#![allow(dead_code, unused_variables)]
//! 3D Object-Centric Physics Benchmark Environment
//!
//! Structured scene: ground, walls, target, obstacle cube, agent ball, free balls.
//! CLEVR-style rendering with per-object masks for slot evaluation.

use crate::envs::Environment;
use burn::tensor::{backend::Backend, Tensor};
use std::collections::HashMap;

// ═══ Vector Math ═══

#[derive(Clone, Copy, Debug, Default)]
struct V3(f32, f32, f32);

impl V3 {
    fn dot(&self, o: &V3) -> f32 { self.0*o.0 + self.1*o.1 + self.2*o.2 }
    fn len(&self) -> f32 { self.dot(self).sqrt() }
    fn norm(&self) -> V3 { let l=self.len(); if l>1e-8{V3(self.0/l,self.1/l,self.2/l)}else{V3(0.,0.,0.)} }
    fn cross(&self, o: &V3) -> V3 { V3(self.1*o.2-self.2*o.1, self.2*o.0-self.0*o.2, self.0*o.1-self.1*o.0) }
}
use std::ops::{Add, Sub, Mul};
impl Add for V3 { type Output=V3; fn add(self,o:V3)->V3{V3(self.0+o.0,self.1+o.1,self.2+o.2)} }
impl Sub for V3 { type Output=V3; fn sub(self,o:V3)->V3{V3(self.0-o.0,self.1-o.1,self.2-o.2)} }
impl Mul<f32> for V3 { type Output=V3; fn mul(self,s:f32)->V3{V3(self.0*s,self.1*s,self.2*s)} }

// ═══ RNG ═══

struct Rng(u64);
impl Rng {
    fn new(s:u64)->Self{Rng(s.wrapping_add(0x9E3779B97F4A7C15))}
    fn next(&mut self)->u64{let mut x=self.0;x^=x<<13;x^=x>>7;x^=x<<17;self.0=x;x}
    fn f32(&mut self)->f32{(self.next() as f32)/(u64::MAX as f32)}
    fn v3(&mut self,s:f32)->V3{V3((self.f32()-0.5)*s,(self.f32()-0.5)*s,(self.f32()-0.5)*s)}
}

// ═══ Object Types ═══

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum ObjKind { Ball, Cube, Wall, Target, Agent, Ground }

#[derive(Clone,Debug)]
pub struct Obj3D {
    pub id: usize, pub kind: ObjKind,
    pub pos: V3, pub vel: V3,
    pub radius: f32, pub mass: f32,
    pub color: [u8;3],
    pub is_dynamic: bool, pub is_agent: bool,
    // for cubes/walls
    pub half: V3,
}

#[derive(Clone,Debug,serde::Serialize)]
pub struct Obj3DState {
    pub id: usize,
    #[serde(rename="type")] pub kind: String,
    pub position: [f32;3], pub velocity: [f32;3],
    pub radius: f32, pub mass: f32,
    pub color: [u8;3],
    pub is_dynamic: bool, pub is_agent: bool,
}

#[derive(Clone,Debug,serde::Serialize)]
pub struct FrameState { pub step: usize, pub objects: Vec<Obj3DState> }

// ═══ Physics ═══

const G:f32=0.25; const DAMP:f32=0.995; const BNC:f32=0.6;
const WMIN:f32=0.0; const WMAX:f32=100.0;

fn physics(objects:&mut[Obj3D], action:f32){
    for o in objects.iter_mut(){ if o.is_agent{ o.vel.0+=action*0.6; } }
    for o in objects.iter_mut(){
        if !o.is_dynamic{continue;}
        o.vel.1-=G; o.vel=o.vel*DAMP; o.pos=o.pos+o.vel;
    }
    // ground
    for o in objects.iter_mut(){
        if !o.is_dynamic{continue;}
        if o.pos.1-o.radius<WMIN{o.pos.1=WMIN+o.radius;o.vel.1=o.vel.1.abs()*BNC;o.vel.0*=0.9;o.vel.2*=0.9;}
    }
    // boundary
    for o in objects.iter_mut(){
        if !o.is_dynamic{continue;}
        if o.pos.0-o.radius<WMIN{o.pos.0=WMIN+o.radius;o.vel.0=o.vel.0.abs()*BNC;}
        if o.pos.0+o.radius>WMAX{o.pos.0=WMAX-o.radius;o.vel.0=-o.vel.0.abs()*BNC;}
        if o.pos.2-o.radius<WMIN{o.pos.2=WMIN+o.radius;o.vel.2=o.vel.2.abs()*BNC;}
        if o.pos.2+o.radius>WMAX{o.pos.2=WMAX-o.radius;o.vel.2=-o.vel.2.abs()*BNC;}
    }
    // wall (sphere vs AABB) - collect wall data first
    let walls_data: Vec<(V3, V3)> = objects.iter()
        .filter(|o| o.kind == ObjKind::Wall)
        .map(|w| (w.pos, w.half)).collect();
    for o in objects.iter_mut() {
        if !o.is_dynamic{continue;}
        for &(wp, wh) in &walls_data {
            let cx = o.pos.0.clamp(wp.0 - wh.0, wp.0 + wh.0);
            let cy = o.pos.1.clamp(wp.1 - wh.1, wp.1 + wh.1);
            let cz = o.pos.2.clamp(wp.2 - wh.2, wp.2 + wh.2);
            let d=(V3(cx,cy,cz)-o.pos).len();
            if d<o.radius&&d>1e-6{
                let n=(o.pos-V3(cx,cy,cz)).norm();
                o.pos=o.pos+n*(o.radius-d);
                let vn=o.vel.dot(&n); if vn<0.0{o.vel=o.vel-n*(vn*(1.0+BNC));}
            }
        }
    }
    // sphere-sphere
    let n=objects.len();
    let rs:Vec<f32>=objects.iter().map(|o|o.radius).collect();
    let ms:Vec<f32>=objects.iter().map(|o|o.mass).collect();
    for i in 0..n{for j in i+1..n{
        if !objects[i].is_dynamic&&!objects[j].is_dynamic{continue;}
        let d=objects[i].pos-objects[j].pos; let dist=d.len();
        let md=rs[i]+rs[j];
        if dist<md&&dist>1e-6{
            let nv=d*(1.0/dist); let ov=md-dist;
            let(m1,m2)=(ms[i],ms[j]); let mt=m1+m2;
            if objects[i].is_dynamic{objects[i].pos=objects[i].pos+nv*(ov*m2/mt);}
            if objects[j].is_dynamic{objects[j].pos=objects[j].pos-nv*(ov*m1/mt);}
            let v1=objects[i].vel.dot(&nv); let v2=objects[j].vel.dot(&nv);
            if objects[i].is_dynamic&&objects[j].is_dynamic{
                let v1n=(v1*(m1-m2)+2.0*m2*v2)/mt;
                let v2n=(v2*(m2-m1)+2.0*m1*v1)/mt;
                objects[i].vel=objects[i].vel+nv*((v1n-v1)*BNC);
                objects[j].vel=objects[j].vel+nv*((v2n-v2)*BNC);
            }else if objects[i].is_dynamic{objects[i].vel=objects[i].vel-nv*(v1*(1.0+BNC));}
            else{objects[j].vel=objects[j].vel-nv*(v2*(1.0+BNC));}
        }
    }}
}

// ═══ CLEVR Renderer ═══

fn proj(cam:&V3, r:&V3, u:&V3, f:&V3, p:&V3)->(f32,f32,f32){let d=*p-*cam;(d.dot(r),d.dot(u),d.dot(f))}

fn render(objects:&[Obj3D], size:usize)->image::RgbImage{
    let s=size as u32; let hf=s as f32*0.5;
    let cam=V3(45.0,75.0,55.0); let look=V3(50.0,30.0,50.0);
    let up=V3(0.0,1.0,0.0); let fwd=(look-cam).norm();
    let r=fwd.cross(&up).norm(); let uc=r.cross(&fwd).norm();
    let sc=s as f32/55.0;
    let mut img=image::RgbImage::new(s,s);
    for y in 0..s{for x in 0..s{img.put_pixel(x,y,image::Rgb([225,225,228]));}}
    // ground grid
    for gx in(0..=100).step_by(10){for gz in(0..=100).step_by(10){
        let(px,py,_)=proj(&cam,&r,&uc,&fwd,&V3(gx as f32,0.3,gz as f32));
        let sx=(hf+px*sc)as i32; let sy=(hf-py*sc)as i32;
        if sx>=0&&sx<s as i32&&sy>=0&&sy<s as i32{img.put_pixel(sx as u32,sy as u32,image::Rgb([210,210,213]));}
    }}
    // sort by depth
    let mut ids:Vec<usize>=(0..objects.len()).collect();
    ids.sort_by(|&a,&b|{
        let(_,_,za)=proj(&cam,&r,&uc,&fwd,&objects[a].pos);
        let(_,_,zb)=proj(&cam,&r,&uc,&fwd,&objects[b].pos);
        zb.partial_cmp(&za).unwrap_or(std::cmp::Ordering::Equal)
    });
    let ld=V3(0.3,-1.0,-0.2).norm();
    for &idx in &ids{
        let o=&objects[idx]; if o.kind==ObjKind::Ground{continue;}
        let(px,py,_)=proj(&cam,&r,&uc,&fwd,&o.pos);
        let sx=(hf+px*sc)as i32; let sy=(hf-py*sc)as i32;
        let rp=(o.radius*sc).max(10.0)as i32;
        let sh=(-ld.1).max(0.4).min(1.0);
        for dy in -rp..=rp{for dx in -rp..=rp{
            if dx*dx+dy*dy>rp*rp{continue;}
            let ix=sx+dx; let iy=sy+dy;
            if ix<0||iy<0||ix>=s as i32||iy>=s as i32{continue;}
            let dist=((dx*dx+dy*dy)as f32).sqrt()/rp as f32;
            let s=sh*(1.0-dist*0.3);
            img.put_pixel(ix as u32,iy as u32,image::Rgb([
                (o.color[0]as f32*s).min(255.)as u8,
                (o.color[1]as f32*s).min(255.)as u8,
                (o.color[2]as f32*s).min(255.)as u8]));
        }}
        // shadow
        if o.pos.1>2.0{
            let(spx,spy,_)=proj(&cam,&r,&uc,&fwd,&V3(o.pos.0,0.6,o.pos.2));
            let ssx=(hf+spx*sc)as i32; let ssy=(hf-spy*sc)as i32;
            let sr=(rp as f32*0.5)as i32;
            for dy in -sr..=sr{for dx in -sr..=sr{
                if dx*dx+dy*dy>sr*sr{continue;}
                let ix=ssx+dx; let iy=ssy+dy;
                if ix<0||iy<0||ix>=s as i32||iy>=s as i32{continue;}
                let e=img.get_pixel(ix as u32,iy as u32);
                img.put_pixel(ix as u32,iy as u32,image::Rgb([(e[0]as f32*0.82)as u8,(e[1]as f32*0.82)as u8,(e[2]as f32*0.87)as u8]));
            }}
        }
    }
    img
}

pub fn render_mask(objects:&[Obj3D], tid:usize, size:usize)->image::GrayImage{
    let s=size as u32; let hf=s as f32*0.5;
    let cam=V3(45.0,75.0,55.0); let look=V3(50.0,30.0,50.0);
    let up=V3(0.0,1.0,0.0); let fwd=(look-cam).norm();
    let r=fwd.cross(&up).norm(); let uc=r.cross(&fwd).norm();
    let sc=s as f32/55.0;
    let mut mask=image::GrayImage::new(s,s);
    for o in objects{
        if o.id!=tid||o.kind==ObjKind::Ground{continue;}
        let(px,py,_)=proj(&cam,&r,&uc,&fwd,&o.pos);
        let sx=(hf+px*sc)as i32; let sy=(hf-py*sc)as i32;
        let rp=(o.radius*sc).max(10.0)as i32;
        for dy in -rp..=rp{for dx in -rp..=rp{
            if dx*dx+dy*dy>rp*rp{continue;}
            let ix=sx+dx; let iy=sy+dy;
            if ix>=0&&iy>=0&&ix<s as i32&&iy<s as i32{mask.put_pixel(ix as u32,iy as u32,image::Luma([255]));}
        }}
    }
    mask
}

// ═══ Environment ═══

pub struct Object3DWorld {
    pub objects: Vec<Obj3D>,
    step_count: usize, max_steps: usize,
    image_size: usize, image_channels: usize,
    rng: Rng, id_counter: usize,
}

impl Object3DWorld {
    pub fn new(max_steps:usize, _ad:usize, ch:usize, sz:usize, seed:u64, _no:usize)->Self{
        let rng=Rng::new(seed);
        let mut s=Self{objects:Vec::new(),step_count:0,max_steps,image_size:sz,image_channels:ch,rng,id_counter:0};
        s.init_scene(); s
    }
    fn init_scene(&mut self){
        self.objects.clear();
        let mut next_id = 0usize;
        self.id_counter = 0;
        let mut nid = || { let i = next_id; next_id += 1; i };
        // Pre-compute random values
        let target_x = 50.+(self.rng.f32()-0.5)*60.;
        let target_z = 50.+(self.rng.f32()-0.5)*60.;
        let cube_x = 25.+self.rng.f32()*50.;
        let cube_z = 25.+self.rng.f32()*50.;
        let agent_x = 15.+self.rng.f32()*20.;
        let agent_y = 50.+self.rng.f32()*20.;
        let ball_data: Vec<(V3, V3, f32)> = (0..3).map(|_| {
            loop {
                let x = 20.+self.rng.f32()*60.;
                let y = 30.+self.rng.f32()*40.;
                let z = 20.+self.rng.f32()*60.;
                let p = V3(x,y,z);
                if self.objects.iter().any(|o| (p-o.pos).len() < o.radius+5.5) { continue; }
                return (p, self.rng.v3(2.), 3.5+self.rng.f32()*2.);
            }
        }).collect();
        // Build all objects
        let mut objs = Vec::new();
        objs.push(Obj3D{id:nid(),kind:ObjKind::Ground,pos:V3(50.,-0.5,50.),vel:V3(0.,0.,0.),radius:100.,mass:1e9,color:[180;3],is_dynamic:false,is_agent:false,half:V3(100.,0.5,100.)});
        for (p,h) in [(V3(50.,15.,2.),V3(50.,15.,4.)),(V3(50.,15.,98.),V3(50.,15.,4.)),(V3(2.,15.,50.),V3(4.,15.,50.)),(V3(98.,15.,50.),V3(4.,15.,50.))]{
            objs.push(Obj3D{id:nid(),kind:ObjKind::Wall,pos:p,vel:V3(0.,0.,0.),radius:h.0.max(h.2),mass:1e9,color:[145;3],is_dynamic:false,is_agent:false,half:h});
        }
        objs.push(Obj3D{id:nid(),kind:ObjKind::Target,pos:V3(target_x,85.,target_z),vel:V3(0.,0.,0.),radius:4.,mass:1e9,color:[50,220,80],is_dynamic:false,is_agent:false,half:V3(0.,0.,0.)});
        objs.push(Obj3D{id:nid(),kind:ObjKind::Cube,pos:V3(cube_x,5.,cube_z),vel:V3(0.,0.,0.),radius:6.,mass:1e5,color:[180,130,60],is_dynamic:false,is_agent:false,half:V3(5.,5.,5.)});
        objs.push(Obj3D{id:nid(),kind:ObjKind::Agent,pos:V3(agent_x,agent_y,50.),vel:V3(0.,0.,0.),radius:5.,mass:10.,color:[60,200,220],is_dynamic:true,is_agent:true,half:V3(0.,0.,0.)});
        let bc=[[255,80,80],[80,150,255],[255,200,40]];
        for i in 0..3 {
            let (pos, vel, r) = &ball_data[i];
            objs.push(Obj3D{id:nid(),kind:ObjKind::Ball,pos:*pos,vel:*vel,radius:*r,mass:5.,color:bc[i],is_dynamic:true,is_agent:false,half:V3(0.,0.,0.)});
        }
        self.objects = objs;
        self.id_counter = next_id;
    }
    pub fn state(&self)->FrameState{FrameState{step:self.step_count,objects:self.objects.iter().map(|o|Obj3DState{id:o.id,kind:format!("{:?}",o.kind).to_lowercase(),position:[o.pos.0,o.pos.1,o.pos.2],velocity:[o.vel.0,o.vel.1,o.vel.2],radius:o.radius,mass:o.mass,color:o.color,is_dynamic:o.is_dynamic,is_agent:o.is_agent}).collect()}}
    pub fn render_masks(&self)->HashMap<usize,image::GrayImage>{
        let mut m=HashMap::new();
        for o in &self.objects{if o.kind==ObjKind::Ground||o.kind==ObjKind::Wall{continue;}m.insert(o.id,render_mask(&self.objects,o.id,self.image_size));}
        m
    }
    fn to_tensor<B:Backend>(&self)->Tensor<B,3>{
        let img=render(&self.objects,self.image_size); let s=self.image_size;
        let mut d=Vec::with_capacity(3*s*s);
        for c in 0..3{for y in 0..s{for x in 0..s{d.push(img.get_pixel(x as u32,y as u32)[c]as f32/255.);}}}
        Tensor::<B,1>::from_floats(d.as_slice(),&Default::default()).reshape([3,s,s])
    }
    fn reward(&self)->f32{
        let a=self.objects.iter().find(|o|o.is_agent);
        let t=self.objects.iter().find(|o|o.kind==ObjKind::Target);
        let(Some(a),Some(t))=(a,t)else{return 0.};
        let d=((a.pos.0-t.pos.0).powi(2)+(a.pos.1-t.pos.1).powi(2)+(a.pos.2-t.pos.2).powi(2)).sqrt();
        (-d*0.03).exp()
    }
}

impl Environment for Object3DWorld {
    fn reset<B:Backend>(&mut self,_:&B::Device)->Tensor<B,3>{self.step_count=0;self.init_scene();self.to_tensor()}
    fn step<B:Backend>(&mut self,a:&[f32],_:&B::Device)->(Tensor<B,3>,f32,bool){
        let act=if a.is_empty(){0.}else{a[0].clamp(-1.,1.)};
        physics(&mut self.objects,act); self.step_count+=1;
        (self.to_tensor(),self.reward(),self.step_count>=self.max_steps)
    }
    fn obs_shape(&self)->[usize;3]{[self.image_channels,self.image_size,self.image_size]}
    fn action_dim(&self)->usize{1}
    fn max_steps(&self)->usize{self.max_steps}
}

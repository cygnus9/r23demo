#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 1014, Nicolas P. Rougier. All rights reserved.
# Distributed under the terms of the new BSD License.
# -----------------------------------------------------------------------------
import sys
import ctypes
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import math
import transforms
import time
import fbo
import random
import argparse
import numpy

import pygame

def start_music():
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load('music.ogg')
    pygame.mixer.music.play()

import assembly.copperbar
import assembly.circles
import assembly.snow
#import assembly.sint
import assembly.matrix
import assembly.particles
import assembly.tree
import assembly.rotate
import assembly.gltf

import geometry.ws2811
import geometry.hub75e

bpm = 140
freq = 250
start = None
lastTime = None
prevPos = None

ps = []
screenWidth = 0
screenHeight = 0

rotationSpeed = 30  # °/s

def interp(t, v0, v1):
    w0 = 1-t
    w1 = t

    return w0 * v0 + w1 * v1

def reltime():
    global start, lastTime, args


    t = pygame.mixer.music.get_pos() / 1000 

    if start is None:
        start = t
        lastTime = start

    dt = t - lastTime
    lastTime = t

    return t, dt
 
def clear():   
    gl.glClearColor(0, 0, 0, 0)    
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)

def do_gltf_depth_buffer(t):
    with depthfbo:
        clear()

        modelview = np.eye(4, dtype=np.float32)
        #transforms.scale(modelview, 5, 5, 5)
        #transforms.xrotate(modelview, 90)
        #transforms.translate(modelview, -25, 32, 0)
        #transforms.scale(modelview, 3, -3, 3)
        gltf.setModelView(modelview)
        gltf.render(t)

def clamp(v, min, max):
    if v < min:
        return min
    if v > max:
        return max
    return v

def inout(v, inp, out, end):
    if v < inp:
        return v / inp
    if v < out:
        return 1
    return 1 - (v - out) / (end-out)

def beat(v, bpm):
    spb = 1.0 / (bpm/60)

    beat = (v % spb) / spb

    return math.pow(1 - beat, 2)

def display():
    global args, mainfbo, texquad, signalgenerator, prevPos

    maint, dt = reltime()

    introtime = 8.0
    switchtime = 6.0

    if maint < introtime:
        t = maint
        clear()
        video.color = (1,1,1,inout(t, 1.0, introtime-1.0, introtime))
        video.render(t)
    else:
        t = maint - introtime

        do_gltf_depth_buffer(t)
        # TODO: move to assembly for "newyear" space effect
        with mainfbo:
            clear()

            # TODO: move to assembly for "newyear" space effect
            if True:
                gltf.apply_animations(t)

            if t > switchtime:
                modelview = np.eye(4, dtype=np.float32)
                transforms.yrotate(modelview, t * 30)
                transforms.translate(modelview, 0, 0, -100)
            else:
                cam = gltf.getNodeModels('Camera')
                modelview = numpy.linalg.inv(cam[0]['model'])

            effect.setModelView(modelview)
            effect.render(t)
            effect.step(t)


        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT| gl.GL_DEPTH_BUFFER_BIT)

        gl.glViewport(0, 0, screenWidth, screenHeight)

        with hbloomfbo:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT| gl.GL_DEPTH_BUFFER_BIT)
            hbloomquad.render()

        h = 4 * beat(t, bpm) + 1
        h = 1
        texquad.color = (0.08 * h, 0.08 * h, 0.08 * h, 1.0)
        texquad.render()
        vbloomquad.color = (0.002, 0.002, 0.002, 1.0)
        vbloomquad.render()

        if t < switchtime:
            effect.setFocus(10  )
            n = int(t * freq) - int((t-dt) * freq)
            for i in range(0, n):
                st = interp(i/n, t-dt, t)
                gltf.apply_animations(st)

                for node in gltf.getNodePositions('Cube'):
                    x,y,z = node['pos']
                    r,g,b,a = node['color']

                    d = 0.01
                    dx = random.uniform(-d, d)
                    dy = random.uniform(-d, d)
                    dz = random.uniform(-d, d)

                    effect.addstar(t, x, y, z, dx, dy, dz, r*2, g*2, b*2)

        else:   
            effect.setFocus(100)
            effect.setGravity(100)
            effect.setDrag(0)
            n = int(t * freq) - int((t-dt) * freq)
            a = 0.4 * t * 2 * math.pi
            l = 5 * math.sin (t * 2 * math.pi * .3) + 8
            b = 0
            l2 = 0

            nx = math.sin(a) * l * 1.5
            ny = math.cos(a) * l * 1.5
            nz = math.cos(a) * l2 * 1.5
            
            for i in range(0, n):
                st = i/n
                
                sx = sy = sz = 0
                if not prevPos:
                    prevPos = (nx,ny,nz)
                    
                sx = nx - prevPos[0]
                sy = ny - prevPos[1]
                sz = nz = prevPos[2]
                
                    
                dx = sx/dt + random.uniform(-1, 1)
                dy = sy/dt + random.uniform(-1, 1)
                dz = sz/dt + random.uniform(-1, 1)
                
                (x,y,z) = (interp(st, prevPos[0], nx), interp(st, prevPos[1], ny), interp(st, prevPos[2], nz))
                
                r,g,b = random.choice([(1.5, 1.8, 0.3), (1.5, 1.3, 0.8)])

                effect.addstar(t, x, y, z, dx/5, dy/5, dz/5, r, g, b)

            prevPos = (nx,ny,nz)

    glut.glutSwapBuffers()
    glut.glutPostRedisplay()


def reshape(width,height):
    global screenWidth, screenHeight
    
    print( width, height )
    
    screenWidth = width
    screenHeight = height
    
    gl.glViewport(0, 0, width, height)
    global effect

    aspect = np.eye(4, dtype=np.float32)
    transforms.scale(aspect, height/width, 1, 1)
    effect.setAspect(aspect)
    gltf.setAspect(aspect)
    hbloomquad.blurvector = (0.1 * height/width, 0)

def keyboard( key, x, y ):
    if key == b'\033':
        glut.glutLeaveMainLoop()
    if key == b' ':
        print('%d', pygame.mixer.music.get_pos())

parser = argparse.ArgumentParser(description='Amazing WS2811 VGA driver')
parser.add_argument('--music', action='store_const', const=True, help='Sync to music')
parser.add_argument('--fullscreen', action='store_true', help='Fullscreen mode')

args = parser.parse_args()

# GLUT init
# --------------------------------------

glut.glutInit()
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
glut.glutCreateWindow(b'Amazing ws2811 VGA renderer')
glut.glutReshapeWindow(960,540)

glut.glutReshapeFunc(reshape)
glut.glutDisplayFunc(display)
glut.glutKeyboardFunc(keyboard)

# Primary offscreen framebuffer
X, Y = 1920, 1080
mainfbo = fbo.FBO(X, Y)
depthfbo = fbo.FBO(X, Y)
hbloomfbo = fbo.FBO(512, 512)


# Emulation shader
dquad = geometry.simple.texquad()
dquad.setTexture(depthfbo.getDepthTexture())

texquad = geometry.simple.texquad()
texquad.setTexture(mainfbo.getTexture())
hbloomquad = geometry.simple.blurtexquad(gain = 5, blurvector = (0.1, 0))
hbloomquad.setTexture(mainfbo.getTexture())
vbloomquad = geometry.simple.blurtexquad(gain = 5, blurvector = (0, 0.1))
vbloomquad.setTexture(hbloomfbo.getTexture())

gl.glEnable(gl.GL_FRAMEBUFFER_SRGB)

start_music()


# Effect
import assembly.newyear
import assembly.video
video = assembly.video.video("fm.avi")

import pyrr

aspect = np.eye(4, dtype=np.float32)
projection = pyrr.matrix44.create_perspective_projection(15, 1.0, 1, 1000)

gltf = assembly.gltf.gltf('text.gltf')
gltf.setProjection(projection)
gltf.color = (5,5,5,1)

effect = assembly.newyear.newyear(depthfbo.getDepthTexture(), rotationSpeed)
effect.setProjection(projection)
effect.setAspect(aspect)

if args.fullscreen:
    glut.glutFullScreen()
glut.glutMainLoop()

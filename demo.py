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

import pygame

def start_music():
	pygame.init()
	pygame.mixer.init()
	pygame.mixer.music.load('loreen.mp3')
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

start = time.time()
lastTime = 0

ps = []
screenWidth = 0
screenHeight = 0

def reltime():
    global start, lastTime, args

    if args.music:
         t = ((pygame.mixer.music.get_pos()-1100) / 454.3438)
    else:
        t = time.time() - start

    lastTime = t

    return t
 
def clear():   
    gl.glClearColor(0, 0, 0, 0)    
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)

def display():
    global args, mainfbo, texquad, signalgenerator

    t = reltime()

    with depthfbo:
        clear()

        modelview = np.eye(4, dtype=np.float32)
        transforms.scale(modelview, 300, 300, 300)
        transforms.yrotate(modelview, t*30)
        transforms.translate(modelview, 0, -10, -100)
        gltf.setModelView(modelview)
        gltf.render(t)

    with mainfbo:
        clear()

        gltf.render(t)

        modelview = np.eye(4, dtype=np.float32)
        transforms.yrotate(modelview, t*30)
        #transforms.translate(modelview, 0, -.03, -.5)
        transforms.translate(modelview, 0, 0, -100)
        effect.setModelView(modelview)
        effect.render(t)
        effect.step(t)


    gl.glClearColor(0, 0, 0, 0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT| gl.GL_DEPTH_BUFFER_BIT)

    gl.glViewport(0, 0, screenWidth, screenHeight)

    with hbloomfbo:
        gl.glClear(gl.GL_COLOR_BUFFER_BIT| gl.GL_DEPTH_BUFFER_BIT)
        hbloomquad.render()

    texquad.color = (0.08, 0.08, 0.08, 1.0)
    texquad.render()
#    vbloomquad.color = (0.002, 0.002, 0.002, 1.0)
#    vbloomquad.render()

#    dquad.render()
#    dquad.color = (0.1, 0.1, 0.1, 1.0)

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
mainfbo = fbo.FBO(960, 540)
depthfbo = fbo.FBO(960, 540)
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

# Effect
import assembly.newyear

effect = assembly.newyear.newyear(depthfbo.getDepthTexture())
import pyrr

aspect = np.eye(4, dtype=np.float32)
projection = pyrr.matrix44.create_perspective_projection(20, 1.0, .5, 1000)

effect.setProjection(projection)
effect.setAspect(aspect)

gltf = assembly.gltf.gltf('test.glb')
gltf.setProjection(projection)
gltf.color = (5,5,5,1)

if args.music:
	start_music()

if args.fullscreen:
    glut.glutFullScreen()
glut.glutMainLoop()

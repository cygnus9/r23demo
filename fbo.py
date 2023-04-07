#!/usr/bin/python

from OpenGL.GL import *
from OpenGL.GL.EXT.framebuffer_object import *

class FBO:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.tex = glGenTextures(1)
        self.prevBuffer = 0

        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glGenerateMipmap(GL_TEXTURE_2D)

        self.fbo = glGenFramebuffers(1)
        
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.fbo)
        
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, self.tex, 0)
        
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0)
        
    def __enter__(self):
        glPushAttrib(GL_VIEWPORT_BIT)
        self.prevBuffer = glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.fbo)
        glViewport(0, 0, self.width, self.height)
        
    def __exit__(self, type, value, traceback):
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.prevBuffer)
        glPopAttrib()
        
    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self.tex)

    def getTexture(self):
        return self.tex
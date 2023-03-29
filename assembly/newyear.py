import assembly
import random
import geometry
import geometry.simple
import OpenGL.GL as gl
import numpy as np
import transforms
import math
from datetime import datetime

class newyear(assembly.assembly):
    freq = 140
    life = 4

    class Stars(geometry.base):
        primitive = gl.GL_QUADS
        srcblend = gl.GL_SRC_ALPHA
        dstblend = gl.GL_ONE

        instanceAttributes = { 'center' : 3, 'color' : 4 }
        attributes = { 'position' : 2 }

        vertex_code = """
            uniform mat4 modelview;
            uniform mat4 projection;
            uniform vec4 objcolor;

            in highp vec4 color;
            in highp vec2 position;
            in highp vec3 center;
            out highp vec4 v_color;
            out highp vec2 v_texcoor;
            void main()
            {
                highp vec4 worldPos = modelview * vec4(center, 1.0);
                highp float scale = abs((worldPos.z + 100.0) * 0.2);
                scale = max(scale, .5);
                gl_Position = (projection * worldPos) + vec4(position, 0.0, 0.0) * scale * 2.0;
                highp float brightness = 1.0/pow(scale, 4.0);
                v_color =  objcolor * color * vec4(vec3(brightness), 1.0);
                v_texcoor = position;
            } 
        """

        fragment_code = """
            in highp vec4 v_color;
            in highp vec2 v_texcoor;
            out highp vec4 f_color;

            void main()
            {
                f_color = vec4(clamp(v_color.rgb * v_color.a * pow((1.0 - length(v_texcoor)),0.25), 0.0, 10.0), 1.0);
            } """

        def __init__(self):
            self.starcolor = (1,1,1,1)
            self.positions = []
            self.colors = []

            super(newyear.Stars, self).__init__()
    
        def getVertices(self):
            verts = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]
            colors = [self.starcolor, self.starcolor, self.starcolor, self.starcolor]

            return { 'position' : verts, 'color' : colors }
            
        def setStars(self, positions, colors):
            self.positions = positions
            self.colors = colors
            self.reloadInstanceData()

        def getInstances(self):
            return { 'center' : self.positions, 'color' : self.colors }

    class AnimatedStar:
        def __init__(self, start, x, y, z, dx, dy, dz, life, basecolor):
            self.start = start
            self.x = x
            self.y = y
            self.z = z
            self.dx = dx
            self.dy = dy
            self.dz = dz
            self.life = life
            self.basecolor = basecolor
            self.shift = random.uniform(0,1)

        def step(self, t, dt):
            self.x = self.x + self.dx * dt
            self.y = self.y + self.dy * dt
            self.z = self.z + self.dz * dt
            slowfactor = (0.5 * dt)
            self.dx *= 1 - (slowfactor * abs(self.dx))
            self.dy *= 1 - (slowfactor * abs(self.dy))
            self.dz *= 1 - (slowfactor * abs(self.dz))
            #self.dy -= 15 * dt

            reltime = t - self.start
            alpha = math.fabs((self.life)-reltime) 
            alpha *= ((1 + (math.sin((reltime + self.shift) * 10))) / 10.0)
            self.color = self.basecolor + (alpha,)

    def __init__(self):
        self.stars = []
        self.geometry = newyear.Stars()
        self.digits = []
        #for i in range(0, 10):
        self.last = 0
        self.lastx = self.lasty = 0
        self.lastz = 0

    def addstar(self, t, dx, dy, dz):
        while self.stars and t-self.stars[0].start > self.life:
            self.stars = self.stars[1:]

        x, y, z = self.getCenter(t)

        w = 5

        a = random.uniform(0, math.pi * 2)
        dx += w*math.cos(a)
        dy += w*math.sin(a)
        dz += w*math.sin(a+math.pi)
        
        color = random.choice([(8.0,6.4,1.6), (8,7.2,4.8)])

        self.stars.append(newyear.AnimatedStar(t, x, y, z, dx, dy, dz, self.life, color))

    def getCenter(self, t):
        t = t * 4
        a = math.sin(0.11 * t * 2 * math.pi) * .3 * math.pi + math.sin(0.13 * t * 2 * math.pi) * .5 * math.pi
        l = math.sin(0.07 * t * 2 * math.pi) * 12

        b = math.sin(0.09 * t * 2 * math.pi) * .2 * math.pi + math.sin(0.19 * t * 2 * math.pi) * .5 * math.pi
        l2 = math.cos(0.03 * t * 2 * math.pi) * 12

        mx = math.sin(a) * l * 1.5
        my = math.cos(a) * l * 1.5
        mz = math.cos(b) * l2 * 1.5

        return mx, my, mz

    def render(self, t):
        dt = t - self.last
        x,y,z = self.getCenter(t)

        if int(t*self.freq) > int(self.last*self.freq) and dt > 0:
            dx = (x - self.lastx)/dt
            dy = (y - self.lasty)/dt
            dz = (z - self.lastz)/dt

            self.addstar(t, dx, dy, dz)

        self.lastx = x
        self.lasty = y
        self.lastz = z

        self.last = t

        positions = []
        colors = []
        for star in self.stars:
            star.step(t, dt)
            positions.append((star.x, star.y, star.z))
            colors.append(star.color)

        self.geometry.setStars(positions, colors)
        self.geometry.render()

        now = datetime.now()

    def setProjection(self, M):
        self.geometry.setProjection(M)

    def setModelView(self, M):
        self.geometry.setModelView(M)


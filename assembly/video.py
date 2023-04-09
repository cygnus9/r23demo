import geometry
import geometry.simple
import numpy as np
import OpenGL.GL as gl
import time
import transforms
from ffpyplayer.player import MediaPlayer

class video(geometry.simple.texquad):
    fragment_code = """
        uniform sampler2D tex;
        out highp vec4 f_color;
        in highp vec2 v_texcoor;
        in highp vec4 v_color;
        
        void main()
        {
            f_color = textureLod(tex, v_texcoor * vec2(-1, 1), 0.0) * v_color;
        } """
        
    def __init__(self, filename):
        self.filename = filename
        self.player = MediaPlayer(self.filename, loop=0)

        while True:
            frame, val = self.player.get_frame()
        
            if frame:
                img, t = frame
                size = img.get_size()
                break
        
        self.w = size[0]
        self.h = size[1]
        self.tex = gl.glGenTextures(1)
        self.n = 0
        
        super(video, self).__init__()

    def setAspect(self, M):
        self.aspect = M

    def getVertices(self):
        aspect = 1 #float(self.h)/float(self.w)
#        verts = [(-1, +aspect), (+1, +aspect), (+1, -aspect), (-1, -aspect)]
        verts = [(-1/aspect, 1), (+1/aspect, 1), (+1/aspect, -1), (-1/aspect, -1)]
        coors = [(1, 0), (0, 0), (0, 1), (1, 1)]
        
        return { 'position' : verts, 'texcoor' : coors }

    def draw(self):
        loc = gl.glGetUniformLocation(self.program, "tex")
        gl.glUniform1i(loc, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        
        geometry.base.draw(self)
        
    def render(self, t):
        frame, val = self.player.get_frame()

        if frame:
            img, t = frame

            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            data = img.to_bytearray()[0]
            size = img.get_size()
            data = bytes(data)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, size[0], size[1], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)

            time.sleep(val)

        super(video, self).render()

    def step(self, t):
        pass

    def changeModelview(self, t):
        modelview = np.eye(4, dtype=np.float32)
        transforms.yrotate(modelview, 180)
        transforms.translate(modelview, 0, 0, -8.0)
        self.setModelView(modelview)

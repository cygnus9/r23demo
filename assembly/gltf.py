import assembly
import random
import geometry
import geometry.simple
import OpenGL.GL as gl
import numpy as np
import transforms
import math
import fbo
import struct
import numpy
from io import BytesIO
from PIL import Image

from pygltflib import GLTF2, Scene

def slerp(t, qa, qb):
    qm = (0,0,0,1)

    # Calculate angle between them.
    cosHalfTheta = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]

    if (abs(cosHalfTheta) >= 1.0):
        return qa

    # Calculate temporary values.
    halfTheta = math.acos(cosHalfTheta)
    sinHalfTheta = math.sqrt(1.0 - cosHalfTheta*cosHalfTheta)

    # if theta = 180 degrees then result is not fully defined
    # we could rotate around any axis normal to qa or qb
    if (abs(sinHalfTheta) < 0.001):
        return (qa[0] * 0.5 + qb[0] * 0.5, qa[1] * 0.5 + qb[1] * 0.5, qa[2] * 0.5 + qb[2] * 0.5, qa[3] * 0.5 + qb[3] * 0.5)

    ratioA = math.sin((1 - t) * halfTheta) / sinHalfTheta
    ratioB = math.sin(t * halfTheta) / sinHalfTheta

    # calculate Quaternion.
    return (qa[0] * ratioA + qb[0] * ratioB,
            qa[1] * ratioA + qb[1] * ratioB,
            qa[2] * ratioA + qb[2] * ratioB,
            qa[3] * ratioA + qb[3] * ratioB)

def interp(t, v0, v1):
    w0 = 1-t
    w1 = t

    if (type(v0) == float):
        return w0 * v0 + w1 * v1

    if len(v0) == 1:
        return (interp(t, v0[0], v1[0]),)

    if len(v0) == 2:
        return (interp(t, v0[0], v1[0]),
                interp(t, v0[1], v1[1]))

    if len(v0) == 3:
        return (interp(t, v0[0], v1[0]),
                interp(t, v0[1], v1[1]),
                interp(t, v0[2], v1[2]))

    if len(v0) == 4:
        return slerp(t, v0, v1)

class Mesh(geometry.base):
    primitive = gl.GL_TRIANGLES
    srcblend = gl.GL_SRC_ALPHA
    dstblend = gl.GL_ONE_MINUS_SRC_ALPHA


    vertex_code = """
        uniform mat4 modelview;
        uniform mat4 projection;
        uniform vec4 objcolor;
        uniform mat4 aspect;

        in highp vec4 color;
        in highp vec3 position;
        in highp vec3 normal;
        in highp vec2 texcoors0;

        out highp vec4 v_color;
        out highp vec2 v_texcoors0;
        void main()
        {
            vec4 worldPos = modelview * vec4(position,1.0);
            vec4 center = modelview * vec4(0.0, 0.0, 0.0, 1.0);
            vec4 worldNormal = modelview * vec4(normal, 1.0) - center;
            gl_Position = aspect * projection * worldPos;
            vec4 direction = vec4(0.0,0.0,1.0,1.0);
            highp float c = pow((1.0 - dot(direction.xyz, normalize(worldNormal.xyz))), 4.0) * 2.0;
            v_color = vec4(c * 1.0, c * 0.9, c * 0.1, 1.0);
            v_texcoors0 = texcoors0;
        }
    """

    fragment_code = """
        in highp vec4 v_color;
        in highp vec2 v_texcoors0;
        out highp vec4 f_color;
        uniform sampler2D texture0;

        void main()
        {
            f_color = v_color;
        #ifdef TEXTURE0
            f_color *= texture(texture0, v_texcoors0);
        #endif
        }
    """

    def __init__(self, mode, vertices, normals, indices, images):
        self.attributes = { 'position' : 3, 'normal': 3 }

        self.primitive = mode
        self.vertices = vertices
        self.normals = normals
        self.indices = indices
        self.textures = []

        for i, image in enumerate(images):
            self.defines += "#define TEXTURE%d" % i
            self.textures.append((self.loadImage(image[0]), image[1]))
            self.attributes['texcoors%d' % i] = 2
        self.aspect = np.eye(4, dtype=np.float32)

        super().__init__()

    def loadImage(self, im):
        imdata = im.tobytes()
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, im.size[0], im.size[1], 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, imdata)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        return tex

    def getVertices(self):
        vertAttribs = { 'position' : self.vertices, 'normal': self.normals }
        for i, texture in enumerate(self.textures):
            vertAttribs['texcoors%d' % i] = texture[1]

        return vertAttribs

    def getTextures(self):
        textures = {}
        for i, texture in enumerate(self.textures):
            textures['texture%d' % i] = texture[0]

        return textures

    def getInstances(self):
        return { }

    def getUniforms(self):
        return { 'aspect': self.aspect }

    def draw(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDrawElements(self.primitive, len(self.indices), gl.GL_UNSIGNED_INT, self.indices)
        gl.glDisable(gl.GL_DEPTH_TEST)

class gltf(assembly.assembly):
    def __init__(self, filename, nodeName = None):
        self.gltf = gltf = GLTF2().load(filename)
        self.modelview = np.eye(4, dtype=np.float32)

        if nodeName:
            self.node = self.getNodeByName(nodeName)
            mesh = gltf.meshes[self.node.mesh]
            primitive = mesh.primitives[0]

            vertices = self.readFromAccessor(gltf, primitive.attributes.POSITION)
            normals = self.readFromAccessor(gltf, primitive.attributes.NORMAL)
            indices = self.readFromAccessor(gltf, primitive.indices)

            images = []

            for image in gltf.images[:1]: # only one texture for now
                texcoors = self.readFromAccessor(gltf, primitive.attributes.TEXCOORD_0)

                try:
                    bufferView = gltf.bufferViews[image.bufferView]
                    print(bufferView)
                    buffer = gltf.buffers[bufferView.buffer]
                    data = gltf.get_data_from_buffer_uri(buffer.uri)[bufferView.byteOffset:bufferView.byteOffset+bufferView.byteLength]
                    stream = BytesIO(data)
                    im = Image.open(stream, formats=['png'])
                    images.append((im, texcoors))
                except Exception as e:
                    print("Failed to load %s: %s" % (buffer.uri[:80], e))
                    pass

            self.geometry = Mesh(primitive.mode, vertices, normals, indices, images)
        else:
            self.geometry = None

        self.animations = []
        for animation in gltf.animations:
            # Meh, only one for now
            animSamplers = []
            for sampler in animation.samplers:
                input = self.readFromAccessor(gltf, sampler.input)
                output = self.readFromAccessor(gltf, sampler.output)

                if len(input) != len(output):
                    raise RuntimeError("Wrong size!")

                animSampler = []
                for inp,out in zip(input, output):
                    animSampler.append((inp, out))

                animSamplers.append(animSampler)

            self.animations.append(animSamplers)


    def getNodeByName(self, name):
        found = [node for node in self.gltf.nodes if node.name == name]
        if found:
            return found[0]

        return None

    def getFormat(self, componentType, type):
        formats = { 5123: "H", 5126: "f" }
        types = { "SCALAR": 1, "VEC2":2, "VEC3": 3, "VEC4": 4 }

        if componentType not in formats:
            raise RuntimeError('Unknown format componentType %d' % componentType)

        if type not in types:
            raise RuntimeError('Unknown format type %s' % type)

        f = formats.get(componentType)
        n = types.get(type)

        return "<" + (f * n)

    def readFromAccessor(self, gltf, accessorId):
        out = []

        # get the binary data for this mesh primitive from the buffer
        accessor = gltf.accessors[accessorId]

        format = self.getFormat(accessor.componentType, accessor.type)
        size = struct.calcsize(format)

        bufferView = gltf.bufferViews[accessor.bufferView]
        buffer = gltf.buffers[bufferView.buffer]
        data = gltf.get_data_from_buffer_uri(buffer.uri)

        # pull each vertex from the binary buffer and convert it into a tuple of python floats
        for i in range(accessor.count):
            index = bufferView.byteOffset + accessor.byteOffset + i*size  # the location in the buffer of this vertex
            d = data[index:index+size]
            v = struct.unpack(format, d)
            out.append(v)

        return out

    def apply_animations(self, t):
        for animId, anim in enumerate(self.gltf.animations):
            for channel in anim.channels:
                nodeid = channel.target.node
                targetNode = self.gltf.nodes[nodeid]

                val = self.interpolateAnim(animId, channel.sampler, t)
                setattr(targetNode, channel.target.path, val)

    def interpolateAnim(self, animId, sampler, t):
        animSampler = self.animations[animId][sampler]

        t = t % animSampler[-1][0][0]

        nextKeyFrame = prevKeyFrame = None
        for i in range(0, len(animSampler)):
            keyFrame = animSampler[i]

            if keyFrame[0][0] > t:
                nextKeyFrame = keyFrame
                prevKeyFrame = animSampler[i-1]
                break

        if nextKeyFrame:
            t0 = prevKeyFrame[0][0]
            t1 = keyFrame[0][0]
            v0 = prevKeyFrame[1]
            v1 = keyFrame[1]

            if t0 == t1:
                return v0

            dt = (t - t0) / (t1-t0)

            ret = interp(dt, v0, v1)
            return ret

        return self.animation[animId][sampler][0][1]

    def getModel(self, node):
        model = np.eye(4, dtype=np.float32)
        if node.scale:
            transforms.scale(model, *node.scale)
        if node.rotation:
            transforms.rotateQ(model, *node.rotation)
        if node.translation:
            transforms.translate(model, *node.translation)

        return model

    def getRecursiveNodeModels(self, nodeId, prefix):
        node = self.gltf.nodes[nodeId]

        if node.name.startswith(prefix):
            return [ { 'model' : self.getModel(node), 'node' : node } ]

        children = []
        for child in node.children:
            children.extend(self.getRecursiveNodeModels(child, prefix))

        return [ { 'model' : np.dot(child['model'], self.getModel(node)), 'node' : child['node'] } for child in children]

    def getNodeModels(self, prefix):
        models = []
        for node in self.gltf.scenes[0].nodes:
            models.extend(self.getRecursiveNodeModels(node, prefix))

        return models

    def getNodePositions(self, prefix):
        models = self.getNodeModels(prefix)

        points = []
        for model in models:
            pos = (0,0,0,1)
            modelview = np.dot(model['model'], self.modelview)

            v = np.dot(pos, modelview)
            point = {
                'pos' : (v[0]/v[3], v[1]/v[3], v[2]/v[3]),
                'node' : model['node'] }

            try:
                point['color'] = self.gltf.materials[self.gltf.meshes[model['node'].mesh].primitives[0].material].pbrMetallicRoughness.baseColorFactor
            except:
                point['color'] = (1,1,1,1)

            points.append(point)

        return points

    def getNodes(self):
        return self.gltf.nodes

    def render(self, t):
        self.apply_animations(t)

        view = np.eye(4, dtype=np.float32)
        model = self.getModel(self.node)

        self.geometry.time = t
        self.geometry.color = self.color
        self.geometry.modelview = np.dot(model, self.modelview)
        self.geometry.render()

    def setProjection(self, M):
        self.geometry.setProjection(M)

    def setModelView(self, M):
        self.modelview = M

    def setAspect(self, M):
        self.geometry.aspect = M


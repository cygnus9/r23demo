uniform mat4 modelview;
uniform mat4 projection;
uniform mat4 aspect;
uniform vec4 objcolor;
uniform float time;
uniform float lifetime;

uniform sampler2D positionTex;
uniform sampler2D colorTex;
uniform sampler2D velocityTex;

in highp vec2 position;
in highp vec2 texcoor;
out highp vec4 v_color;
out highp vec2 v_texcoor;
void main()
{
    highp vec4 postex = texelFetch(positionTex, ivec2(texcoor), 0);
    highp vec3 center = postex.xyz;
    highp vec4 color = vec4(texelFetch(colorTex, ivec2(texcoor), 0).rgb, 1.0);
    highp vec4 velocity = texelFetch(velocityTex, ivec2(texcoor), 0);

    highp vec4 projectedCenter = aspect * projection * modelview * vec4(center, 1.0);
    highp float scale = abs((projectedCenter.z - 100.0) * 0.2);
    scale = max(scale, .4);

    highp vec4 projectedVelocity = aspect * projection * modelview * vec4(velocity);
    highp vec2 velocity_2d = projectedVelocity.xy;

    highp float angle = atan(velocity_2d.y, velocity_2d.x);
    highp mat2x2 rotation = mat2x2(cos(angle), -sin(angle), sin(angle), cos(angle));

    highp vec2 transformed_position = 
        vec2(position.x * max(1., length(velocity_2d) / 40.), 0) * rotation +
        vec2(0, position.y) * rotation;

    gl_Position = projectedCenter + vec4(transformed_position, 0.0, 0.0) * scale * 2.0 * aspect;
   
    highp float brightness = 1.0/pow(scale, 2.0);
    brightness *= 100.0 / projectedCenter.z;
    highp float sparkle = (1.0 + sin((time - postex.w) * 10.0)) / 2.0;
    highp float fade = clamp(1.0 - ((time - postex.w) / lifetime), 0.0, 1.0);
    v_color =  objcolor * color * vec4(vec3(brightness * sparkle * fade), 1.0);
    v_texcoor = position;
} 
uniform vec2 resolution;
attribute vec3 center;
attribute float radius;
varying vec3 v_center;
varying float v_radius;
void main()
{
    v_radius = radius;
    v_center = center;
    gl_PointSize = 2.0 + ceil(2.0*radius);
    gl_Position = vec4(2.0*center.xy/resolution-1.0, v_center.z, 1.0);
}

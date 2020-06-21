varying vec3 v_center;
varying float v_radius;

void main()
{
    vec2 p = (gl_FragCoord.xy - v_center.xy)/v_radius;
    float z = 1.0 - length(p);
    if (z < 0.0) discard;

    gl_FragDepth = 0.5*v_center.z + 0.5*(1.0 - z);

    vec3 color = vec3(1.0, 1.0, 1.0);
    vec3 normal = normalize(vec3(p.xy, z));
    vec3 direction = vec3(0.0, 0.0, 1.0);
    float diffuse = max(0.0, dot(direction, normal));
    float specular = pow(diffuse, 24.0);
    vec4 phong = vec4(max(diffuse*color, specular*vec3(1.0)), 1.0);
    gl_FragColor = 1*phong;
}

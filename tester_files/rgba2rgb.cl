

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void rgba_to_rgb(__read_only image2d_t rgba, __global uchar* rgb)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int w = get_image_width(rgba);
    
    int4 p = read_imagei(rgba, sampler, pos);
    rgb[(pos.x + pos.y * w)*3 + 0] = p.x;
    rgb[(pos.x + pos.y * w)*3 + 1] = p.y;
    rgb[(pos.x + pos.y * w)*3 + 2] = p.z;
}

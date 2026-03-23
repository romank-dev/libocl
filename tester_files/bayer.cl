

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void bayer_to_rgb(__read_only image2d_t rggb, __write_only image2d_t rgba)
{
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    
    int2 T = pos + (int2)(0,1);
    int2 B = pos - (int2)(0,1);
    int2 L = pos - (int2)(1,0);
    int2 R = pos + (int2)(1,0);
    int2 TL = pos + (int2)(-1,1);
    int2 BL = pos - (int2)(1,1);
    int2 TR = pos + (int2)(1,1);
    int2 BR = pos - (int2)(-1,1);
    
    int p_X = read_imagei(rggb, sampler, pos).x;
    int p_T = read_imagei(rggb, sampler, T).x;
    int p_B = read_imagei(rggb, sampler, B).x;
    int p_L = read_imagei(rggb, sampler, L).x;
    int p_R = read_imagei(rggb, sampler, R).x;
    int p_TR = read_imagei(rggb, sampler, TR).x;
    int p_TL = read_imagei(rggb, sampler, TL).x;
    int p_BR = read_imagei(rggb, sampler, BR).x;
    int p_BL = read_imagei(rggb, sampler, BL).x;
    
    int odd_x = pos.x % 2;
    int odd_y = pos.y % 2;
    
    int red =   odd_x*odd_y*p_X +
                !odd_x*odd_y*(p_L+p_R)/2 + 
                odd_x*!odd_y*(p_T+p_B)/2 + 
                !odd_x*!odd_y*(p_TL+p_TR+p_BL+p_BR)/4;
    
    int blue =  !odd_x*!odd_y*p_X + 
                odd_x*!odd_y*(p_L+p_R)/2 +
                !odd_x*odd_y*(p_T+p_B)/2 +
                odd_x*odd_y*(p_TL+p_TR+p_BL+p_BR)/4;
                
    int green = (odd_x^odd_y)*p_X + 
                !(odd_x^odd_y)*(p_T+p_B+p_L+p_R)/4;
                
    
    write_imagei(rgba, pos, (int4)(red,green,blue,255));
}

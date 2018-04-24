#ifndef __DEFAULTS_H__
#define __DEFAULTS_H__


namespace defaults {
    constexpr unsigned int num_threads_in_calc_sqnorm_func = 1024;

    constexpr unsigned int num_threads_in_twotops_func = 64; // must be 2 to the power of n.
    constexpr unsigned int num_blocks_y_in_twotops_func = 2048;
//    constexpr unsigned int num_threads_in_gather_func = 128;
    constexpr unsigned int num_threads_in_swap_sort_func = 128;
}


#endif // __DEFAULTS_H__


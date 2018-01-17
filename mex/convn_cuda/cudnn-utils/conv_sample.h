#ifndef __CUDNNUTILS_H__
#define __CUDNNUTILS_H__


namespace cudnnutils {

template <typename T_ELEM>
int conv_handler(T_ELEM* hostI, T_ELEM* hostF, T_ELEM* hostO, int algo, int* dimA, int* filterdimA, int benchmark);

}

#endif

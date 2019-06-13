#ifndef _MEASURE_H_
#define _MEASURE_H_

#include <sys/time.h>
#include <stdlib.h>

long long mseconds() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec*1000 + t.tv_usec/1000;
}

#endif

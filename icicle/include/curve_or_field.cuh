#pragma once
#ifndef CURVE_OR_FIELD_H
#define CURVE_OR_FIELD_H

#ifdef CURVE_ID
#include "curves/curve_config.cuh"
using namespace curve_config;
#else
#include "fields/field_config.cuh"
using namespace field_config;
#endif

#endif
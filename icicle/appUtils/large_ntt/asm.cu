/*

Copyright (c) 2023 Yrrid Software, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the �Software�), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED �AS IS�, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("subc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("subc.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("sub.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint64_t mulwide(uint32_t a, uint32_t b) {
  uint64_t r;
  
  asm volatile ("mul.wide.u32 %0,%1,%2;" : "=l"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint64_t madwide(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r;

  asm volatile ("mad.wide.u32 %0,%1,%2,%3;" : "=l"(r) : "r"(a), "r"(b), "l"(c));
  return r;
}

__device__ __forceinline__ uint64_t madwide_cc(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r;

  asm volatile ("{\n\t"
                ".reg .u32 lo,hi;\n\t"
                "mov.b64        {lo,hi},%3;\n\t"
                "mad.lo.cc.u32  lo,%1,%2,lo;\n\t"
                "madc.hi.cc.u32 hi,%1,%2,hi;\n\t"
                "mov.b64        %0,{lo,hi};\n\t"
                "}" : "=l"(r) : "r"(a), "r"(b), "l"(c));
  return r;
}

__device__ __forceinline__ uint64_t madwidec_cc(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r;

  asm volatile ("{\n\t"
                ".reg .u32 lo,hi;\n\t"
                "mov.b64        {lo,hi},%3;\n\t"
                "madc.lo.cc.u32 lo,%1,%2,lo;\n\t"
                "madc.hi.cc.u32 hi,%1,%2,hi;\n\t"
                "mov.b64        %0,{lo,hi};\n\t"
                "}" : "=l"(r) : "r"(a), "r"(b), "l"(c));
  return r;
}

__device__ __forceinline__ uint64_t madwidec(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r;

  asm volatile ("{\n\t"
                ".reg .u32 lo,hi;\n\t"
                "mov.b64        {lo,hi},%3;\n\t"
                "madc.lo.cc.u32 lo,%1,%2,lo;\n\t"
                "madc.hi.u32    hi,%1,%2,hi;\n\t"
                "mov.b64        %0,{lo,hi};\n\t"
                "}" : "=l"(r) : "r"(a), "r"(b), "l"(c));
  return r;
}

__device__ __forceinline__ uint32_t ulow(uint64_t wide) {
  uint32_t r;

  asm volatile ("mov.b64 {%0,_},%1;" : "=r"(r) : "l"(wide));
  return r;
}

__device__ __forceinline__ uint32_t uhigh(uint64_t wide) {
  uint32_t r;

  asm volatile ("mov.b64 {_,%0},%1;" : "=r"(r) : "l"(wide));
  return r;
}

__device__ __forceinline__ uint64_t make_wide(uint32_t lo, uint32_t hi) {
  uint64_t r;
  
  asm volatile ("mov.b64 %0,{%1,%2};" : "=l"(r) : "r"(lo), "r"(hi));
  return r;
}
  
__device__ __forceinline__ uint32_t prmt(uint32_t lo, uint32_t hi, uint32_t c) {
  uint32_t r;
    
  asm volatile ("prmt.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(c));
  return r;
}

__device__ __forceinline__ uint32_t load_shared_u32(uint32_t addr) {
  uint32_t r;
  
  asm volatile ("ld.shared.u32 %0,[%1];" : "=r"(r) : "r"(addr)); 
  return r;
}

__device__ __forceinline__ void store_shared_u32(uint32_t addr, uint32_t value) {
  asm volatile ("st.shared.u32 [%0],%1;" : : "r"(addr), "r"(value));
}

__device__ __forceinline__ uint64_t load_shared_u64(uint32_t addr) {
  uint64_t r;
  
  asm volatile ("ld.shared.u64 %0,[%1];" : "=l"(r) : "r"(addr)); 
  return r;
}

__device__ __forceinline__ void store_shared_u64(uint32_t addr, uint64_t value) {
  asm volatile ("st.shared.u64 [%0],%1;" : : "r"(addr), "l"(value));
}

__device__ __forceinline__ void bar(uint32_t name, uint32_t count) {
  asm volatile ("bar.sync %0,%1;" : : "r"(name), "r"(count));
}

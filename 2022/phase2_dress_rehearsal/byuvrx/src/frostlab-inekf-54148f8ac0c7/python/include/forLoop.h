#ifndef FOR_LOOP
#define FOR_LOOP

#include <type_traits>
#include <pybind11/pybind11.h>
#include "globals.h"

namespace py = pybind11;
using namespace pybind11::literals;


/*  Compile time for loops, adapted from 
        https://artificial-mind.net/blog/2020/10/31/constexpr-for
    This is what requires C++17, it is theoretically doable without it, 
    but is just miles harder without "if constexpr" */
template <int i, int I, class F>
constexpr void forConstSingle(F&& f){
    if constexpr (i < I){
        f(std::integral_constant<decltype(i), i>());
        forConstSingle<i+1, I>(f);
    }
}

template <int i, int I, int j, int J, class F>
constexpr void forConstDouble(F&& f){
    if constexpr(i < I){
        forConstSingle<j,J>( [&f](auto js){f(std::integral_constant<decltype(i), i>(),js);} );
        forConstDouble<i+1,I,j,J>(f);
    }
}

template <int i, int I, int j, int J, int k, int K, class F>
constexpr void forConstTriple(F&& f){
    if constexpr(i < I){
        forConstDouble<j,J,k,K>( [&f](auto js, auto ks){f(std::integral_constant<decltype(i), i>(),js,ks);} );
        forConstTriple<i+1,I,j,J,k,K>(f);
    }
}

// For loops to automatically iterate over our needed values
template <class F>
constexpr void forSO(F&& f){
    forConstSingle<-1,AUG+1>(f);
}

template <class F>
constexpr void forSE(F&& f){
    forConstDouble<1,COL+1,-1,AUG+1>(f);

    // Use a seperate for loop to skip 0 for columns
    forConstSingle<-1,AUG+1>([&f](auto j){ f(std::integral_constant<int, -1>(),j); });
}

template <class F>
constexpr void forSOVec(F&& f){
    forConstDouble<-1,AUG+1,1,VEC+1>(f);
    
    // Handle when VEC=-1
    forConstSingle<-1,AUG+1>([&f](auto i){ f(i,std::integral_constant<int, -1>()); });
}

template <class F>
constexpr void forSEVec(F&& f){
    forConstTriple<1,COL+1,-1,AUG+1,1,VEC+1>(f);

    // Use a seperate for loop to skip 0 for vectors/columns
    forConstDouble<-1,AUG+1,1,VEC+1>([&f](auto j, auto k){ f(std::integral_constant<int, -1>(),j,k); });
    forConstDouble<1,COL+1,-1,AUG+1>([&f](auto i, auto j){ f(i,j,std::integral_constant<int, -1>()); });

    // Handle case when COL=-1 and VEC=-1
    forConstSingle<-1,AUG+1>([&f](auto j){ f(std::integral_constant<int, -1>(),j,std::integral_constant<int, -1>()); });
}


// Helpers for elsewhere
template <int C, int A>
std::string makeNameSE(){
    std::string name = "_";
    name += C == Eigen::Dynamic ? "D" : std::to_string(C);
    name += "_";
    name += A == Eigen::Dynamic ? "D" : std::to_string(A);
    return name;
}

template <int A>
std::string makeNameSO(){
    std::string name = "_";
    name += A == Eigen::Dynamic ? "D" : std::to_string(A);
    return name;
}

#endif // FOR_LOOP
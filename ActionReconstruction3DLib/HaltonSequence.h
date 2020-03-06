#pragma once

#include <array>
#include <numeric>

namespace ar3d
{
    namespace utils
    {
        template<int N>
        std::array<double, N> halton(int i)
        {
            static_assert(N <= 6, "halton sequence only supported until dimension 6");
            static_assert(N >= 1, "Dimension must be >= 1");
            static const int npvec[] = { 2,    3,    5,    7,   11,   13 };
            int d;
            int i1;
            int j;
            std::array<double, N> prime_inv;
            std::array<double, N> r;
            std::array<int, N> t;

            for (j = 0; j < N; j++)
            {
                t[j] = i;
            }
            //
            //  Carry out the computation.
            //
            for (j = 0; j < N; j++)
            {
                prime_inv[j] = 1.0 / (double)(npvec[j]);
            }

            for (j = 0; j < N; j++)
            {
                r[j] = 0.0;
            }

            while (0 < std::accumulate(t.begin(), t.end(), 0))
            {
                for (j = 0; j < N; j++)
                {
                    d = (t[j] % npvec[j]);
                    r[j] = r[j] + (double)(d)* prime_inv[j];
                    prime_inv[j] = prime_inv[j] / (double)(npvec[j]);
                    t[j] = (t[j] / npvec[j]);
                }
            }

            return r;
        }
    }
}
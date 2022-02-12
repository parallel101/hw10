#include <cstdio>
#include <vector>
#include <limits>
#include <cstdlib>
#include "ticktock.h"

constexpr int N = 1<<12;

// TODO: 改造成小彭老师说的稀疏数据结构
std::vector<char> cells(N * N);
std::vector<char> outcells(N * N);

void init() {
#pragma omp parallel for collapse(2)
    for (int y = N / 2 - 1024; y < N / 2 + 1024; y++) {
        for (int x = N / 2 - 1024; x < N / 2 + 1024; x++) {
            cells[x * N + y] = std::rand() % 2;
        }
    }
    std::swap(cells, outcells);
}

void step() {
    // 这里遍历 [1, N-1] 区间是为了避免访问越界
#pragma omp parallel for collapse(2)
    for (int y = 1; y < N-1; y++) {
        for (int x = 1; x < N-1; x++) {
            int neigh = 0;
            neigh += cells[x * N + (y + 1)];
            neigh += cells[x * N + (y - 1)];
            neigh += cells[(x + 1) * N + (y + 1)];
            neigh += cells[(x + 1) * N + y];
            neigh += cells[(x + 1) * N + (y + 1)];
            neigh += cells[(x - 1) * N + (y + 1)];
            neigh += cells[(x - 1) * N + y];
            neigh += cells[(x - 1) * N + (y - 1)];
            if (cells[x * N + y]) {
                if (neigh == 2 || neigh == 3) {
                    outcells[x * N + y] = 1;
                } else {
                    outcells[x * N + y] = 0;
                }
            } else {
                if (neigh == 3) {
                    outcells[x * N + y] = 1;
                } else {
                    outcells[x * N + y] = 0;
                }
            }
        }
    }
    std::swap(cells, outcells);
}

void calc_bound() {
    int rightbound = std::numeric_limits<int>::min();
    int leftbound = std::numeric_limits<int>::max();
#pragma omp parallel for collapse(2) reduction(max:rightbound) reduction(min:leftbound)
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (cells[x * N + y]) {
                rightbound = std::max(rightbound, x);
                leftbound = std::min(leftbound, x);
            }
        }
    }
    printf("left=%d, right=%d\n", leftbound, rightbound);
}

int main() {
    TICK(main);

    init();
    for (int times = 0; times < 1024; times++) {
        printf("step %d\n", times);
        step();
    }
    calc_bound();

    TOCK(main);
    return 0;
}

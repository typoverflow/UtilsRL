#ifndef UTILSRL_DATA_STRUCTURE_SUMTREE
#define UTILSRL_DATA_STRUCTURE_SUMTREE

#include <cmath>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>

class SumTree {
public: 
    SumTree(int max_size): max_size(max_size), valid_size(0), curr(0) {
        if (max_size <= 0) throw std::invalid_argument("`max_size` of the sum tree cannot be zero!");
        tree_depth = ceil(log2(this->max_size)); 
        tree_size = pow(2, tree_depth+1) - 1;
        node_size = pow(2, tree_depth) - 1;
        tree_body.assign(tree_size, 0);
    }
    SumTree& update(int idx, double new_value) {
        int tidx = _get_tree_idx(idx);
        double diff = new_value - tree_body[tidx];
        tree_body[tidx] += diff;
        for (; (tidx-1) >= 0; ){
            tidx = _father(tidx);
            tree_body[tidx] += diff;
        }
        return *this;
    }
    SumTree& add(double new_value) {
        update(curr, new_value);
        curr = (curr+1) % max_size;
        valid_size = std::min(valid_size+1, max_size);
        return *this;
    }
    int find(double target, bool scale=true) {
        if (scale) target *= total();
        int tidx;
        double tvalue;
        tidx = _find_helper(target, 0, tvalue);
        return _get_idx(tidx);
    }
    void show() {
        for (int d=0; d<=tree_depth; ++d) {
            printf("[Depth %d]: ", d);
            for (int i=0; i<std::pow(2, d); ++i)
                printf("%.3f  ", tree_body[i+std::pow(2, d)-1]);
            printf(" \n");
        }
    }

private:
    int max_size; 
    int tree_depth; 
    int tree_size;
    int node_size;
    int valid_size = 0;
    int curr = 0;
    std::vector<double> tree_body;

    inline int _left(const int tree_idx) {return 2*tree_idx+1;}
    inline int _right(const int tree_idx) {return 2*tree_idx+2;}
    inline int _father(const int tree_idx) {return (tree_idx-1)/2;}
    inline int _get_tree_idx(const int idx) {return node_size + idx;}
    inline int _get_idx(const int tidx) {return tidx - node_size;}
    inline double total() {return tree_body[0];}

    int _find_helper(double target, int tidx, double& value) {
        if (_left(tidx) >= tree_size) {
            value = tree_body[tidx];
            return tidx;
        }
        double left_value = tree_body[_left(tidx)];
        if (target <= left_value) return _find_helper(target, _left(tidx), value);
        else return _find_helper(target - left_value, _right(tidx), value);
    }
};

#endif
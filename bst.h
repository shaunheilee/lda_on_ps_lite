#include <iostream>
#include <vector>
#include <sstream>
#include <string>

using namespace std;

void build_bst(vector<double>& sl){
    size_t sz = sl.size();

    size_t i = 0, j = sz;
    while(1){
        if(i + 1 == sl.size()) break;
        double s = sl[i] + sl[i+1];
        sl.push_back(s);
        i += 2;
        j += 1;
    }
}

size_t parent_index(vector<double>& sl, size_t idx){
    size_t n_leaf = (sl.size() + 1) / 2;
    size_t p = idx / 2;
    return p + n_leaf;
}

pair<int, int> child_index(vector<double>& sl, size_t idx){
    size_t sz = sl.size();
    return make_pair<int, int>(2*idx - sz - 1, 2*idx - sz);
}

void update_bst(vector<double>& sl, size_t idx, double delta){
    size_t sz = sl.size();
    size_t p = idx;
    while(p < sz){
        sl[p] += delta;
        p = parent_index(sl, p);
    }
}

template<typename T>
void debug_bst(vector<T>& sl){
    stringstream ss;
    ss<<"size="<<sl.size()<<" -> ";
    for(auto it:sl)
        ss<<it<<" ";
    ss<<endl;
    cerr<<ss.str();
}


template<typename T>
void debug_vec(vector<T>& sl, string d){
    cerr<<"DEBUG VEC:"<<d<<" ";
    debug_bst(sl);
}

size_t search_bst(vector<double>& sl, double r, int idx = -1){
    int root = idx;
    if(root == -1) root = sl.size() - 1;
    auto childs = child_index(sl, root);
    if(childs.first >= 0){
        if(sl[childs.first] > r)
            return search_bst(sl, r, childs.first);
        else
            return search_bst(sl, r - sl[childs.first], childs.second);
    }
    return root; 
}

#include <iostream>
#include <string>
#include <unordered_map>
#include "dmlc/io.h"
#include "dmlc/logging.h"
#include "base/localizer.h"
#include <sstream>
#include <string>

using namespace std;
using namespace dmlc;
typedef uint64_t K;

class Dump {
 
 public:

  Dump(string file_in, string file_out, bool need_inverse) : file_in_(file_in),file_out_(file_out), need_inverse_(need_inverse) {}
  ~Dump() {data_.clear();}

  // value type stored on sever nodes, can be also other Entrys

  struct WordTopicEntry{
      std::vector<int> vec;
      inline void Load(Stream *fi) {
          // TODO increasemental training ?
          fi->Read(&vec);
      }
      inline void Save(Stream *fo) {
          // key has just be saved in kv server
          fo->Write(vec);
      }
      inline bool Empty() const {return vec.empty();}
  };


  void LoadModel(const std::string filename) {
    Stream* fi = CHECK_NOTNULL(Stream::Create(filename.c_str(), "r"));
    K key;
    while (true) {
      if (fi->Read(&key, sizeof(K)) != sizeof(K)) break;
      data_[key].Load(fi);
    }
    cout << "loaded " << data_.size() << " kv pairs\n";
  }

  // how to dump the info
  void DumpModel(const std::string filename) {
    Stream* fo = CHECK_NOTNULL(Stream::Create(filename.c_str(), "w"));
    dmlc::ostream os(fo);
    int dumped = 0;
    for (const auto& it : data_) {
      if (it.second.Empty()) continue;
      uint64_t feature_id = need_inverse_ ? ReverseBytes(it.first) : it.first;

      std::stringstream ss;
      ss << feature_id << '\t';
      auto vec = it.second.vec;
      for(int i = 0; i < vec.size(); i++)
        ss << vec[i] << "\t";
      std::string s = ss.str();
      s[s.length() - 1] = '\n';
      os << s;
      dumped ++;
    }
    cout << "dumped " << dumped << " kv pairs\n";
  }

  void run() {
    LoadModel(file_in_);
    DumpModel(file_out_);
  }

 private:
  unordered_map<K, WordTopicEntry> data_;
  string file_in_;
  string file_out_;
  bool need_inverse_;
};

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "Usage: <model_in> <dump_out> [need_inverse]\n";
    return 0;
  }
  google::InitGoogleLogging(argv[0]);
  string model_in, dump_out;
  bool need_inverse = false;
  for (int i = 1; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      if (!strcmp(name, "model_in")) model_in = val;
      if (!strcmp(name, "dump_out")) dump_out = val;
      if (!strcmp(name, "need_inverse")) need_inverse = !strcmp(val, "0") ? false : true;
    }
  }
  Dump d(model_in, dump_out, need_inverse);
  d.run();
  return 0;
}

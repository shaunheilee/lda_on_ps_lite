/**
 * @file   lda.h
 * @brief  lda.
 */
#include "solver/minibatch_solver.h"
#include "config.pb.h"
#include "progress.h"
#include "base/localizer.h"
#include <set>
#include <algorithm>
#include <sstream>
#include <stdlib.h>
#include <functional>
#include <thread>
#include "bst.h"

#define BUF_SIZE 1024000
using namespace std;
namespace dmlc {
namespace lda {

using FeaID = ps::Key;
template <typename T> using Blob = ps::Blob<T>;

/**
 * \brief the base handle
 */
struct BaseHandle {
 public:
  BaseHandle() { ns_ = ps::NodeInfo::NumServers(); }
  inline void Start(bool push, int timestamp, int cmd, void* msg) { }

  inline void Finish() {
    // avoid too frequently reporting
    ++ ct_;
    if (ct_ >= ns_ && reporter) {
      Progress prog; prog.new_w() = new_w; reporter(prog);
      new_w = 0; ct_ = 0;
    }
  }

  inline static void Update(float cur_w, float old_w) {
    if (old_w == 0 && cur_w != 0) {
      ++ new_w;
    } else if (old_w != 0 && cur_w == 0) {
      -- new_w;
    }
  }

  void Load(Stream* fi) { }
  void Save(Stream *fo) const { }

  // learning rate
  float alpha = 0.1, beta = 1;

  std::function<void(const Progress& prog)> reporter;
  static int64_t new_w;

 private:
  int ct_ = 0;
  int ns_ = 0;
};

struct WordTopicEntry{
    std::vector<int> vec;
    inline void Load(Stream *fi) {
        // TODO increasemental training ?
        fi->Read(&vec);
    }
    inline void Save(Stream *fo) const {
        // key has just be saved in kv server
        fo->Write(vec);
    }
    inline bool Empty() const {return vec.empty();}
};

struct WordTopicHandle: public BaseHandle{
    public:
        inline void Start(bool push, int ts, int cmd, void * msg){}

        inline void Push(FeaID key, Blob<const int> vec, WordTopicEntry& wt){
            std::vector<int> & old_wt = wt.vec;
            if(wt.Empty()) old_wt.resize(vec.size);
            for(unsigned int i = 0; i < vec.size; i++){
                old_wt[i] += vec[i];
            }
        }

        inline void Pull(FeaID key, const WordTopicEntry& wt, Blob<int>& send){
            if(wt.vec.size() != 0){
                send.data = (int*)(wt.vec.data());
                send.size = wt.vec.size();
            }
            else{
                send.data[0] = 0;
            }
        }
};

class LDAServer : public solver::MinibatchServer {
 public:
  LDAServer(const Config& conf) : conf_(conf) {
      CreateServer<WordTopicEntry, WordTopicHandle>();
  }
  virtual ~LDAServer() { }

  virtual void ProcessRequest(ps::Message* request){
    if (request->task.msg().size() == 0) return;
    dmlc::solver::IterCmd cmd(request->task.cmd());
    auto filename = ModelName(request->task.msg(), cmd.iter());
    if (cmd.save_model()) {
      Stream* fo = CHECK_NOTNULL(Stream::Create(filename.c_str(), "w"));
      SaveModel(fo);
      delete fo;
    } else if (cmd.load_model()) {
      Stream* fi = CHECK_NOTNULL(Stream::Create(filename.c_str(), "r"));
      LoadModel(fi);
      delete fi;
    }
  }

  template <typename Entry, typename Handle>
  void CreateServer() {
    Handle h;

    h.reporter = [this](const Progress& prog) {
      ReportToScheduler(prog.data);
    };
    ps::OnlineServer<int, Entry, Handle> s(h, conf_.topic_num());
    server_ = s.server();
  }

  virtual void LoadModel(Stream* fi) {
    server_->Load(fi);
    Progress prog; prog.new_w() = BaseHandle::new_w; ReportToScheduler(prog.data);
    BaseHandle::new_w = 0;
  }

  virtual void SaveModel(Stream* fo) const {
    server_->Save(fo);
  }

  Config conf_;
  ps::KVStore* server_;

  private:
    std::string ModelName(const std::string& base, int iter){
        std::string name = base + "/word_topic/";
        if (iter >=0) name += "_iter-" + std::to_string(iter);
        return name + "_part-" + std::to_string(ps::NodeInfo::MyRank());
    }
};

typedef std::vector<FeaID> Doc;
typedef std::unordered_map<FeaID, std::vector<int> > FV;
typedef unordered_map<FeaID, int> NZM;
typedef unordered_map<FeaID, NZM> FNZ;

class LDAWorker : public solver::MinibatchWorker {
 public:
  LDAWorker(const Config& conf) : conf_(conf) {
    mb_size_       = conf_.minibatch();
    concurrent_mb_ = conf_.max_concurrency();
    nt_            = conf_.num_threads();
    model_out_     = conf_.model_out();
    tk_            = conf_.topic_num();
    alpha          = conf_.alpha();
    beta           = conf_.beta();
    words_num      = conf_.words_num();
    niter          = conf_.max_data_pass();

    srand(123);
  }
  virtual ~LDAWorker() {
  }

 protected:
  unsigned int  tk_;
  unsigned int  num_files = 0;
  unsigned int  words_num;
  float alpha, beta;
  std::hash<std::string> hash_;
  int  niter;

  // parse input line into doc
  void str2doc(char * strbuf, Doc& d){
    char * p = strbuf;
    int i = 0;
    while(*p != '\0'){
        char * h = p;
        while((*p != '\t' && *p != ' ') && *p != '\0') ++p;
        if(*p == '\t' || *p == ' '){
            *p = '\0';
            ++p;
        }
        if(i == 0)
            d.push_back((atoll(h)));
        else
            d.push_back(ReverseBytes(atoll(h)));
        d.push_back(0);
        ++i;
    }
  }

  // push word topic to server
  // will output feaids in ascend order
  void pushWordTopic(FV& wt){
    std::vector<FeaID> feaids;
    for(auto it = wt.begin(); it != wt.end(); ++it)
        feaids.push_back(it->first);

    sort(feaids.begin(), feaids.end());
    std::vector<int> vals;
    for(unsigned int i = 0; i< feaids.size(); i++){
        FeaID id = feaids[i];
        vals.insert(vals.end(), wt[id].begin(), wt[id].end());
    }
    int ts = kv_.Push(feaids, vals); 
    kv_.Wait(ts);
  }

  // pull word topics from server
  // feaids are the specific word ids that will be pulled
  // feaids will be sorted afterward
  void pullWordTopic(set<FeaID>& ids, FNZ& wt){
    vector<FeaID> feaids;
    for(auto it: ids)
        feaids.push_back(it);

    sort(feaids.begin(), feaids.end());
    std::vector<int> vals;
    int ts = kv_.Pull(feaids, &vals);
    kv_.Wait(ts);

    for(size_t i = 0; i < feaids.size(); i++){
        if(feaids[i] == 77240) cerr<<"77240 in pull sets!"<<endl;
        auto start = i * tk_;
        for(auto j = start; j < start + tk_; ++j){
            if(vals[j] != 0)
                wt[feaids[i]][j - start] = vals[j];
            else
                wt[feaids[i]].erase(j - start);
        }
    }
  }

  void initial(std::string filename, FNZ& words, FV& wt, FNZ& dnz){
    dmlc::Stream *fi = dmlc::Stream::Create(filename.c_str(), "r");
    dmlc::istream is(fi);
    wt[0].resize(tk_);
    
    file_word_ids[filename].insert(0);
    while(!is.eof()){
        char buf[BUF_SIZE];
        Doc d;
        // load a doc
        is.getline(buf, BUF_SIZE);
        if(strlen(buf) == 0)
            break;
        // to doc
        str2doc(buf, d);
        // random topic
        FeaID docid = -1;
        for(size_t i = 0 ; i < d.size(); i+=2){
            if(i == 0){
                // do nothing, this is doc id
                docid = d[i];
            }
            else{
                d[i+1] = rand() % tk_;
                if(wt[d[i]].empty()){
                    wt[d[i]].resize(tk_);
                }
                wt[d[i]][d[i+1]] += 1;
                // word id=0 is a special topic
                // sum over all words for every topic
                // TPW
                wt[0][d[i+1]] += 1;
                // fill words
                words[d[i]][docid] = d[i+1];
                // fill dnz
                dnz[docid][d[i+1]] += 1;
                
                file_word_ids[filename].insert(d[i]);
            }
        }
    }
    
    // first time, push full word vecs to server
    pushWordTopic(wt);

    delete fi;
  }

  /*
  // save sampled tokens to file
  void saveDocs(std::string filename, std::vector<Doc>& docs){
    dmlc::Stream *fo = dmlc::Stream::Create(filename.c_str(), "w");
    for(size_t i = 0; i < docs.size(); i++){
        std::stringstream ss;
        Doc & d = docs[i];
        for(size_t j = 0; j < d.size(); j += 2)
            ss<<ReverseBytes(d[j])<<"\t"<<d[j+1]<<"\t";
        std::string s = ss.str();
        s[s.length()-1] = '\n';
        fo->Write(s.c_str(), s.length());
    }
    delete fo;
  }
  */

  // save doc topics
  // doc topic will always on work node
  // in order to save time of network transfering
  void saveDocTopic(std::string& filename, FNZ& dnz){
    dmlc::Stream *fo = dmlc::Stream::Create(filename.c_str(), "w");
    for(auto it = dnz.begin(); it != dnz.end(); ++it){
        std::stringstream ss;
        ss<<it->first<<"\t";
        auto v = it->second;
        for(auto nzit : it->second)
            if(nzit.second != 0)
                    ss<<nzit.first<<":"<<nzit.second<<"\t";
        std::string s = ss.str();
        s[s.length()-1] = '\n';
        fo->Write(s.c_str(), s.length());
    }
    delete fo;
  }

  static void fill_nz(FV& d, FNZ& nz){
    for(auto t: d){
        for(size_t i = 0; i < t.second.size(); ++i)
            if(t.second[i] != 0)
                nz[t.first][i] += t.second[i];
    }
  }

  inline void update(NZM& dti, NZM& wtj, NZM& tpw, map<int, double>& tpw_vb, 
       vector<int>& wv1, vector<int>& wv2, size_t k, int v){
      dti[k] += v; wtj[k] += v; tpw[k] += v; wv1[k] += v; wv2[k] += v; tpw_vb[k] += v;
  }

  void sparse_gibbs_sampling(FNZ& words, FNZ& wnz, FNZ& dnz, FV& wt_diff){
    NZM& tpw = wnz[0];
    double vbeta = words_num * beta, bucket_smooth = 0;
    map<int, double> tpw_vb;
    vector<double> arr_bucket_smooth;
    arr_bucket_smooth.resize(tk_);

    for(auto it: tpw){
        tpw_vb[it.first] = vbeta + it.second;
        double t = alpha * beta / tpw_vb[it.first];
        arr_bucket_smooth[it.first] = t;
        bucket_smooth += t;
    }
    build_bst(arr_bucket_smooth);
    double tstart = GetTime();

    float adnz =0, ct = 0;
    for(auto wit : wnz){
        NZM& tokens = words[wit.first];
        FeaID wid = wit.first;
        NZM& wtj = wnz[wid];
        // get non zero topic for this document

        if(wt_diff[0].empty())
            wt_diff[0].resize(tk_);

        double bucket_word = 0;
        // for Ftree searching
        vector<double > arr_bucket_word;
        arr_bucket_word.resize(tk_);

        for(auto it: wtj){
            double t = alpha * it.second / tpw_vb[it.first];
            arr_bucket_word[it.first] = t;
            bucket_word += t;
        }

        build_bst(arr_bucket_word);
    
        if(wt_diff[wid].empty())
            wt_diff[wid].resize(tk_);

        // loop through the doc tokens
        for(auto tit:tokens){
            FeaID did = tit.first;
            unsigned int _k = tit.second;
            NZM& dti = dnz[did];
            ct+=1;
            adnz += dti.size();
        
            update(dti, wtj, tpw, tpw_vb, wt_diff[0], wt_diff[wid], _k, -1);
            // update bucket_smooth
            double delta = 0;
            delta = -alpha * beta * (1 / (tpw_vb[_k] + 1) - 1 / tpw_vb[_k]);
            bucket_smooth = bucket_smooth + delta;
            update_bst(arr_bucket_smooth, _k, delta);
            
            // update bucket_word
            delta = -alpha * ((wtj[_k] + 1) / (tpw_vb[_k] + 1) - wtj[_k] / tpw_vb[_k]);
            bucket_word = bucket_word + delta;
            update_bst(arr_bucket_word, _k, delta);
            
            // remove zero elements
            if(dti[_k] == 0) dti.erase(_k);
            if(wtj[_k] == 0) wtj.erase(_k);

            // calculate bucket_doc
            double bucket_doc = 0;
            for(auto it : dti){
                // operator [] will change the content of wtj
                double v = wtj.find(it.first) == wtj.end() ? 0 : wtj[it.first];
                bucket_doc += it.second * (beta + v) / tpw_vb[it.first];
            }
          
            // cdf
            double cdf = bucket_doc + bucket_word + bucket_smooth;
            // sampling
            double r =  cdf * (rand() % 100000000) / 100000000.0;
            size_t k = 0;
            if(r < bucket_doc){// select in doc bucket
                double t = 0;
                for(auto it: dti){
                    double v = wtj.find(it.first) == wtj.end() ? 0:wtj[it.first];
                    t += it.second * (beta + v)/ tpw_vb[it.first];
                    if(r < t){
                        k = it.first;
                        break;
                    }
                }
            }
            else
            if(r < bucket_doc + bucket_word){// select in word bucket
                k = search_bst(arr_bucket_word, r - bucket_doc);
                if(k<0 || k>=tk_) cerr<<"bucket_doc sampling err!"<<endl;
            }
            else{// select in smooth bucket
                k = search_bst(arr_bucket_smooth, r - bucket_doc - bucket_word); 
                if(k<0 || k>=tk_) {
                    cerr<<"sampling:"<< r - bucket_doc - bucket_word<<endl;
                    debug_bst(arr_bucket_smooth);
                    cerr<<"bucket_smooth sampling err! topic:"<<k<<endl<<endl<<endl;
                }
                //cerr<<"bucket smooth"<<endl; 
            }

            update(dti, wtj, tpw, tpw_vb, wt_diff[0], wt_diff[wid], k, 1);
            // assigning new topic
            tokens[did] = k;
            // update bucket_smooth
            delta = -alpha * beta * (1 / (tpw_vb[k] - 1) - 1 / tpw_vb[k]);
            bucket_smooth = bucket_smooth + delta;
            update_bst(arr_bucket_smooth, k, delta);
            // update bucket_doc
            delta = -alpha * ((wtj[k] - 1) / (tpw_vb[k] - 1) - wtj[k] / tpw_vb[k]);
            bucket_word = bucket_word + delta;
            update_bst(arr_bucket_word, k, delta);
        }
    }
    //
    cerr<<"sampling time:"<<GetTime() - tstart<<" adnz:"<<adnz / ct<<endl;
  }

  FNZ words, dnz;
  //vector<Doc> docs;
  int cur_file_index = 0;
  vector<string> filenames;
  unordered_map<string, set<FeaID> > file_word_ids;
  // override parent's process function
  virtual void Process(const Workload& wl){
    CHECK_GE(wl.file.size(), (size_t)1);
    auto file = wl.file[0];
   
    FNZ wnz;
    FV wt, wt_diff;
   
    cerr<<"DATA PASS:"<<wl.data_pass<<endl; 

    double t2;
    if(wl.data_pass == 0){// for initialize
        t2 = GetTime();
        filenames.push_back(file.filename);
        initial(file.filename, words, wt, dnz);
        fill_nz(wt, wnz);
        cerr<<"initial:"<<GetTime() - t2<<endl;
        if(0){
            int sz = 0;
            for(auto it:dnz){
                sz += it.second.size();
            }
            cerr<<"avg dnz size:"<<sz * 1.0 / dnz.size()<<endl;
        }

        return;
    }
    else{
        t2 = GetTime();
        cur_file_index = cur_file_index % filenames.size();
        string& filename = filenames[cur_file_index];
        // sync word topic
        pullWordTopic(file_word_ids[filename], wnz);
        cerr<<filename<<endl;
    }

    if(0){
        int sz = 0;
        for(auto it:wnz){
            sz += it.second.size();
        }
        cerr<<"before samping avg wnz size:"<<sz * 1.0 / wnz.size()<<endl;
    }
    // do gibbs sampling on docs
    double t3 = GetTime();
    sparse_gibbs_sampling(words, wnz, dnz, wt_diff);
    if(0){
        int sz = 0;
        for(auto it:wnz){
            sz += it.second.size();
        }
        cerr<<"after sampling wnz size:"<<sz * 1.0 / wnz.size()<<endl<<endl<<endl;
    }

    double t4 = GetTime();
    cerr<<"sampling:"<<t4-t3<<endl;
    pushWordTopic(wt_diff); // should push diff rather than wt itself
    ++cur_file_index;

    // it's time to save result
    if(wl.data_pass == niter - 1 && cur_file_index % filenames.size() == 0){
        // save doc topic
        string outputname = model_out_ + "/doc_topic/_part-" + std::to_string(ps::NodeInfo::MyRank());
        cerr<<"saving doc topic:"<<outputname<<" size="<<dnz.size()<<endl;
        saveDocTopic(outputname, dnz);
        cerr<<"done!"<<endl;
    }
  }

  virtual void ProcessMinibatch(const Minibatch& mb, const Workload& wl) {

  }
 private:
  void SetFilters(bool push, ps::SyncOpts* opts) {
    if (conf_.fixed_bytes() > 0) {
      opts->AddFilter(ps::Filter::FIXING_FLOAT)->set_num_bytes(
          conf_.fixed_bytes());
    }
    if (conf_.key_cache()) {
      opts->AddFilter(ps::Filter::KEY_CACHING)->set_clear_cache(push);
    }
    if (conf_.msg_compression()) {
      opts->AddFilter(ps::Filter::COMPRESSING);
    }
  }
  Config conf_;
  int nt_ = 2;
  std::string model_out_;
  ps::KVWorker<int> kv_;
};


/**
 * \brief the scheduler  for LDA 
 */
class LDAScheduler : public solver::MinibatchScheduler {
 public:
  LDAScheduler(const Config& conf) { Init(conf); }
  virtual ~LDAScheduler() { }

  virtual std::string ProgHeader() { return Progress::HeadStr(); }

  virtual std::string ProgString(const solver::Progress& prog) {
    prog_.data = prog;
    return prog_.PrintStr();
  }
 private:
  Progress prog_;
};

}  // namespace lda 
}  // namespace dmlc

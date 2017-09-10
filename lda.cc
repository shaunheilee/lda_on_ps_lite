#include "lda.h"
#include "ps.h"
#include "base/arg_parser.h"

namespace ps {
App* App::Create(int argc, char *argv[]) {
  CHECK_GE(argc, 2) << "\nusage: " << argv[0] << " conf_file";
  ::dmlc::ArgParser parser;
  if (strcmp(argv[1], "none")) parser.ReadFile(argv[1]);
  parser.ReadArgs(argc-2, argv+2);
  ::dmlc::lda::Config conf; parser.ParseToProto(&conf);

  NodeInfo n;
  if (n.IsWorker()) {
    return new ::dmlc::lda::LDAWorker(conf);
  } else if (n.IsServer()) {
    return new ::dmlc::lda::LDAServer(conf);
  } else if (n.IsScheduler()) {
    return new ::dmlc::lda::LDAScheduler(conf);
  } else {
    LOG(FATAL) << "unknown node";
  }
  return NULL;
}
}  // namespace ps

int64_t dmlc::lda::BaseHandle::new_w = 0;

int main(int argc, char *argv[]) {
  return ps::RunSystem(&argc, &argv);
}

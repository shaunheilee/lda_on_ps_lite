include ../../ps-lite/make/ps_app.mk

all: build/lda.dmlc build/dump.dmlc

clean:
	rm -rf build *.pb.*

build/lda.dmlc: build/config.pb.o build/lda.o $(DMLC_SLIB)
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@

build/dump.dmlc: build/dump.o $(DMLC_SLIB)
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@

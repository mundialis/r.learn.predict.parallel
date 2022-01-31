MODULE_TOPDIR = ../..

PGM = r.learn.predict.parallel

# note: to deactivate a module, just place a file "DEPRECATED" in the subdir
ALL_SUBDIRS := ${sort ${dir ${wildcard */.}}}
DEPRECATED_SUBDIRS := ${sort ${dir ${wildcard */DEPRECATED}}}
RM_SUBDIRS := bin/ docs/ etc/ scripts/
SUBDIRS_1 := $(filter-out $(DEPRECATED_SUBDIRS), $(ALL_SUBDIRS))
SUBDIRS := $(filter-out $(RM_SUBDIRS), $(SUBDIRS_1))

# $(warning ALL_SUBDIRS is $(ALL_SUBDIRS))
# $(warning DEPRECATED_SUBDIRS is $(DEPRECATED_SUBDIRS))
# $(warning SUBDIRS is $(SUBDIRS))

include $(MODULE_TOPDIR)/include/Make/Dir.make

default: parsubdirs htmldir

install: installsubdirs
	$(INSTALL_DATA) $(PGM).html $(INST_DIR)/docs/html/

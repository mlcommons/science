

CUDA_VISIBLE_DEVICES ?= 1
EXPERIMENTS = $(foreach exp,$(wildcard dgx/*),$(exp)/nohup.out)


.PHONY: all
all: dgx

.PHONY: run
run: $(EXPERIMENTS)

.PHONY: dgx
dgx: dgx.json
	$(info Experiments generated)

dgx.json:
	cms sbatch generate dgx.in.slurm --setup=dgx-config.yaml --name=dgx

run-%: dgx/card_name_a100_gpu_count_1_cpu_num_1_mem_64GB_TFTTransformerepochs_%/nohup.out
	$(info Experiment launched)

$(EXPERIMENTS): dgx.json
	$(info Running experiment $@)
	(cd $(dir $@) && export CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) && nohup bash slurm.sh &  echo $$! > $(dir $@)/pid.txt)
	sleep 2
	printf -- "Running in process %s" "$(shell cat $(dir $@)pid.txt)"

.PHONY: watch
watch:
	tail -f dgx/*/nohup.out

.PHONY: clean
clean:
	@-rm -rf dgx dgx.json jobs-dgx.sh
	@-rm -f dgx.slurm

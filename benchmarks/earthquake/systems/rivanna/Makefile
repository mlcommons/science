
all: project localscratch shm generate

generate: jobs-project.sh jobs-localscratch.sh jobs-dgx.sh

run: submit

submit: all
	sh jobs-project.sh
	sh jobs-localscratch.sh
	sh jobs-dgx.sh

.PHONY: project
project: project.json

.PHONY: localscratch
localscratch: localscratch.json

.PHONY: dgx
dgx: dgx.json

.PHONY: shm
shm: shm.json

.PHONY: cli
cli:
	cms sbatch generate rivanna-project.in.slurm --config=sbatch-rivanna.yaml --name=earthquake --noos --experiment=\"card_name=a100 gpu_count=1 cpu_num=6 mem=32GB\" --attribute=\"time=12:00:00\" --mode=h --dir=project


jobs-%.sh: %.json
	cms sbatch generate submit --name=$< > $@

%.json: rivanna-%.yaml
	cms sbatch generate rivanna.in.slurm --setup=$< --name=$(basename $@) --noos

dgx.json: rivanna-dgx.yaml
	cms sbatch generate rivanna-dgx.in.slurm --setup=$< --name=$(basename $@) --noos


.PHONY: stop
stop:
	for i in "$$(squeue --user $$USER | awk 'NR>1{print $$1}')"; do scancel $$i ; done

.PHONY: clean
clean:
	@-rm -rf localscratch localscratch.json jobs-localscratch.sh
	@-rm -rf project project.json jobs-project.sh
	@-rm -rf dgx dgx.json jobs-dgx.sh
	@-rm -f rivanna.slurm

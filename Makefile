train:
	@python src/watergenic_coding_test/train.py

train_dynamic:
	@python src/watergenic_coding_test/train_dynamic.py

predict:
	@python src/watergenic_coding_test/predict.py


########## Testing ##########

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

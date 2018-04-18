.PHONY : setup

init:
	git clone https://github.com/AkashGanesan/video-context-transcription	

setup:
	module load python-dev/3.5.2	
	cd video-context-transcription
	virtualenv -p python3 env
	source env/bin/activate
	pip install --upgrade pip
	pip install -r requirements
	python -m spacy download en_core_web_lg
	python -m nltk.downloader all

.PHONY : demo
demo:
	echo "Not Implemented"

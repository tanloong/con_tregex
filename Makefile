.PHONY: refresh clean build release install test cov lint

refresh: lint clean build install

clean:
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf build
	rm -rf dist
	rm -rf src/*.egg-info
	rm -rf htmlcov
	rm -rf coverage.xml
	rm -rf parser.out
	rm -rf src/parser.out
	pip uninstall -y pytregex || true

build:
	python setup.py sdist bdist_wheel

release:
	python -m twine upload dist/*

install:
	python setup.py install

test:
	python -m unittest

cov:
	python -m coverage run -m unittest
	python -m coverage html

lint:
	black src/ tests/ --exclude src/pytregex/ply --line-length 97 --preview
	flake8 src/ tests/ --exclude src/pytregex/ply/ --count --max-line-length=97 --statistics --ignore=E203,E501,W503,F841
	mypy src/ --ignore-missing-imports --exclude src/pytregex/ply/ --exclude src/pytregex/test.py

README_zh_tw.md: README_zh_cn.md
	cd ~/software/zhconv && python -m zhconv zh-tw < ~/projects/pytregex/README_zh_cn.md > ~/projects/pytregex/README_zh_tw.md

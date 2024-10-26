

package:
	mkdir dendrite/src/ndarray
	cp -r ndarray/src/* dendrite/src/ndarray/
	mv dendrite/src/ndarray/lib.rs dendrite/src/ndarray/mod.rs

	mkdir dendrite/src/metrics
	cp -r metrics/src/* dendrite/src/metrics/
	mv dendrite/src/metrics/lib.rs dendrite/src/metrics/mod.rs

	mkdir dendrite/src/clustering
	cp -r clustering/src/* dendrite/src/clustering/
	mv dendrite/src/clustering/lib.rs dendrite/src/clustering/mod.rs

	mkdir dendrite/src/datasets
	cp -r datasets/src/* dendrite/src/datasets/
	mv dendrite/src/datasets/lib.rs dendrite/src/datasets/mod.rs

	mkdir dendrite/src/models
	cp -r models/src/* dendrite/src/models/
	mv dendrite/src/models/lib.rs dendrite/src/models/mod.rs

	mkdir dendrite/src/bayes
	cp -r bayes/src/* dendrite/src/bayes/
	mv dendrite/src/bayes/lib.rs dendrite/src/bayes/mod.rs

	mkdir dendrite/src/autodiff
	cp -r autodiff/src/* dendrite/src/autodiff/
	mv dendrite/src/autodiff/lib.rs dendrite/src/autodiff/mod.rs

	mkdir dendrite/src/preprocessing
	cp -r preprocessing/src/* dendrite/src/preprocessing/
	mv dendrite/src/preprocessing/lib.rs dendrite/src/preprocessing/mod.rs

	mkdir dendrite/src/knn
	cp -r knn/src/* dendrite/src/knn/
	mv dendrite/src/knn/lib.rs dendrite/src/knn/mod.rs

	mkdir dendrite/src/trees
	cp -r trees/src/* dendrite/src/trees/
	mv dendrite/src/trees/lib.rs dendrite/src/trees/mod.rs

	mkdir dendrite/src/regression
	cp -r regression/src/* dendrite/src/regression/
	mv dendrite/src/regression/lib.rs dendrite/src/regression/mod.rs

clean:
	sudo rm -r  dendrite/src/ndarray
	sudo rm -r  dendrite/src/metrics
	sudo rm -r dendrite/src/clustering
	sudo rm -r dendrite/src/datasets
	sudo rm -r dendrite/src/models
	sudo rm -r dendrite/src/bayes
	sudo rm -r dendrite/src/autodiff
	sudo rm -r dendrite/src/preprocessing
	sudo rm -r dendrite/src/knn
	sudo rm -r dendrite/src/trees
	sudo rm -r dendrite/src/regression
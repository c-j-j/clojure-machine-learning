(ns machine-learning.linear-regression
  (:require [incanter.charts :as charts]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.core.matrix :as matrix])
  (:use [incanter.core :only [view]]))

(def alpha 0.01)
(def iterations 1500)

(defn read-csv []
  (with-open [in-file (io/reader "src/machine_learning/ex1_data.csv")]
  (doall
   (csv/read-csv in-file))))

(def data (read-csv))
(def xs (map (comp read-string first) data))
(def y (map (comp read-string second) data))
(def number-of-examples (count xs))

(def X (-> [(repeat number-of-examples 1) xs]
           (matrix/matrix)
           (transpose)))

(def initial-theta [0 0])

(defn hypothesis[X theta]
  (matrix/mmul X theta))

(defn mean-square-error [guess actual]
  (-> (matrix/sub guess actual)
      (matrix/square)
      (matrix/esum)))

(defn compute-cost [X y theta]
  (let [predicted (hypothesis X theta)]
    (/ (mean-square-error predicted y)
       (* 2 number-of-examples))))

(defn cost-derivative [X y theta]
  (let [prediction (hypothesis X theta)]
    (-> prediction
        (matrix/sub y)
        (matrix/mmul X)
        (matrix/mul (/ alpha number-of-examples)))))

(defn gradient-descent [X y theta alpha number-of-iterations]
  (loop [remaining-iterations number-of-iterations
         theta theta]
    (if (zero? remaining-iterations)
      theta
      (recur (dec remaining-iterations)
             (matrix/sub theta (cost-derivative X y theta))))))

(def result (gradient-descent X y initial-theta alpha 1500))

(defn plot-data[x y]
  (let [plot (charts/scatter-plot x y
                                  :x-label "Population of City in 10,000s"
                                  :y-label "Profit in $10,000s")]
    (doto plot
      (charts/add-function #(+ (first result) (* (second result) %)) 0 25)
      view)))

(plot-data xs y)

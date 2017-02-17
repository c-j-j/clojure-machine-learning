(ns machine-learning.multi-linear-regression
  (:require [incanter.charts :as charts]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.core.matrix :as matrix])
  (:use [incanter.core :only [view]]))

(def alpha 0.01)
(def iterations 1500)

(defn read-csv []
  (with-open [in-file (io/reader "src/machine_learning/ex2_data.csv")]
  (doall
   (csv/read-csv in-file))))

(defn convert-to-ints [m]
  (mapv #(mapv read-string %) m))

(def data (convert-to-ints (read-csv)))

(def xs (mapv pop data))

(def y (mapv last data))

(def number-of-examples (count xs))

(defn prefix-one[X]
  (mapv #(cons 1 %) X))

(def X (matrix/matrix xs))

(defn average [numbers]
  (/ (apply + numbers) (count numbers)))

(defn standard-deviation [coll]
  (let [avg (average coll)
        squares (for [x coll]
                  (let [x-avg (- x avg)]
                    (* x-avg x-avg)))
        total (count coll)]
    (-> (/ (apply + squares)
           (- total 1))
        (Math/sqrt))))

(defn normalise-row[x]
  (let [mean (average x)
        std (standard-deviation x)]
    (->> x
         (mapv #(- % mean))
         (mapv #(/ % std)))))

(defn normalise-features[X]
  (->> X
      (matrix/transpose)
      (mapv normalise-row)
      (matrix/transpose)))

(def initial-theta (matrix/transpose (matrix/matrix [0 0 0])))

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

(defn costs[X y theta alpha number-of-iterations]
  (loop [remaining-iterations number-of-iterations
         theta theta
         costs []]
    (if (zero? remaining-iterations)
      costs
      (recur (dec remaining-iterations)
             (matrix/sub theta (cost-derivative X y theta))
             (conj costs (compute-cost X y theta))))))

(def Xa (-> X
            (normalise-features)
           (prefix-one)))

(def result (gradient-descent Xa y initial-theta alpha 1500))
(def cost-history (costs Xa y initial-theta alpha 1500))

(defn plot-cost-history[y]
  (let [plot (charts/scatter-plot (range 0 1500) y
                                  :x-label "Iteration"
                                  :y-label "Cost")]
    (doto plot view)))

(plot-cost-history cost-history)

(defn plot-data[x y]
  (let [plot (charts/scatter-plot x y
                                  :x-label "Population of City in 10,000s"
                                  :y-label "Profit in $10,000s")]
    (doto plot
      (charts/add-function #(+ (first result) (* (second result) %)) 0 25)
      view)))

;; (plot-data xs y)

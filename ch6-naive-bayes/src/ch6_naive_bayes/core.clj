(ns ch6-naive-bayes.core)

(defn make-classifier []
  (atom {:data {} :counter {}}))

(defn increment-feature [classifier feature category]
  (let [result  ((keyword feature) (:data @classifier) {category 0})
        updated (merge-with + {category 1} result)]
    (swap! classifier assoc-in [:data (keyword feature)] updated)))

(defn increment-category [classifier category]
  (let [result  (category (:counter @classifier) 0)]
    (swap! classifier assoc-in [:counter category] (inc result))))

(defn feature-probability [classifier feature category]
  (let [count (category ((keyword feature) (:data @classifier) 0))
        div   (category (:counter @classifier))]
    (if (= div 0)
      0
      (/ count div))))

(defn category-probability [classifier category]
  (let [count (category (:counter @classifier))
        div   (reduce + (vals (:counter @classifier)))]
    (if (= div 0)
      0
      (/ count div))))

(defn weighted-probability [classifier feature category weight assumed-prob]
  (let [basic-probabilty (feature-probability classifier feature category)
        totals           (reduce + (vals (feature @classifier)))]
    (/ (+ (* weight assumed-prob) (* totals basic-probabilty)) (+ weight totals))))

(defn prob-of-category-given-features [classifier category & features]
  (let [category-prob (category-probability classifier category)
        weighted-prob (reduce * 1 (map #(weighted-probability classifier % category 1.0 0.5) features))]
    (* category-prob weighted-prob)))

(defprotocol Feature
  (get-words [str]))

(extend-type String
  Feature
  (get-words [str] (vec (clojure.string/split str #" "))))

(defn train [classifier terms category]
  (increment-category classifier category)
  (doseq [word terms]
    (increment-feature classifier word category)))
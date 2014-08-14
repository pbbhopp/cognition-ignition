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

(defn division-guard [number divisor]
  (if (= divisor 0)
    0
    (/ number divisor)))

(defn feature-probability [classifier feature category]
  (let [count (category ((keyword feature) (:data @classifier) 0))
        div   (category (:counter @classifier))]
    (division-guard count div)))

(defn category-probability [classifier category]
  (let [count (category (:counter @classifier))
        div   (reduce + (vals (:counter @classifier)))]
    (division-guard count div)))

(defn weighted-probability [classifier feature category weight assumed-prob]
  (let [basic-probabilty (feature-probability classifier feature category)
        totals           (reduce + (vals (feature @classifier)))]
    (/ (+ (* weight assumed-prob) (* totals basic-probabilty)) (+ weight totals))))

(defn prob-of-category-given-features [classifier category & features]
  (let [category-prob (category-probability classifier category)
        weight-probs  (map #(weighted-probability classifier % category 1.0 0.5) features)
        weighted-prob (reduce * 1 weight-probs)]
    (* category-prob weighted-prob)))

(defprotocol Feature
  (get-words [str]))

(extend-type String
  Feature
  (get-words [str] (vec (clojure.string/split str #" "))))

(defn train [classifier document category]
  (increment-category classifier category)
  (doseq [word document]
    (increment-feature classifier word category)))
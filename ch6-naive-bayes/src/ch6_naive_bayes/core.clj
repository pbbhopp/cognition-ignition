(ns ch6-naive-bayes.core)

(defn make-classifier []
  (atom {:data {} :counter {}}))

(defn increment-feature [classifier feature category]
  (let [result  ((keyword feature) (:data @classifier) {category 0})
        updated (merge-with + {category 1} result)]
    (swap! classifier assoc-in [:data (keyword feature)] updated)))

(defn increment-category [classifier category]
  (let [result (category (:counter @classifier) 0)]
    (swap! classifier assoc-in [:counter category] (inc result))))

(defn division-guard [number divisor]
  (if (= divisor 0)
    0
    (/ number divisor)))

(defn feature-probability [classifier feature category]
  (let [count (category ((keyword feature) (:data @classifier)) 0)
        div   (category (:counter @classifier))]
    (division-guard count div)))

(defn category-probability [classifier category]
  (let [count (category (:counter @classifier))
        div   (reduce + (vals (:counter @classifier)))]
    (division-guard count div)))

(defn weighted-probability [classifier feature category weight assumed-prob]
  (let [basic-prob (feature-probability classifier feature category)
        totals     (reduce + (vals (feature (:data @classifier))))]
    (/ (+ (* weight assumed-prob) (* totals basic-prob)) (+ weight totals))))

(defn prob-of-category-given-features [classifier category features]
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

(defn sorted-map-by-val [m]
  (into (sorted-map-by (fn [key1 key2] (compare [(get m key2) key2] [(get m key1) key1]))) m))

(defn categories-prob [classifier categories features]
  (let [cat-prob  (fn [cat] (prob-of-category-given-features classifier cat features))
        cat-probs (reduce (fn [m cat] (assoc m cat (cat-prob cat))) {} categories)]
    (sorted-map-by-val cat-probs)))

(defn classify [classifier document]
  (let [features  (map keyword (get-words document))
        all-cats  (keys (:counter @classifier))
        cat-probs (categories-prob classifier all-cats features)]
    (key (first cat-probs))))

# SWE-bench Lite Latency + Accuracy

Tasks: 50 | Accepted: 5 | Rejected: 45

|#|repo|id|lines|ly_total|lr_total|ly_cached|lr_cached|ly_bleu|lr_bleu|matcher|gate|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|0|django|11999|8/12|2595|2085|0|110|0.5833|0.8333|no_overlap|reject|
|1|matplotlib|25442|16/23|2320|2462|0|188|0.2130|0.5185|no_overlap|reject|
|2|django|13315|9/17|2729|2710|0|31|0.1860|0.1977|no_overlap|reject|
|3|sympy|12454|7/7|2041|2154|0|30|0.3250|0.3500|no_overlap|reject|
|4|sympy|23117|9/11|2370|2429|28|28|0.3125|0.2969|span_overlap_high|allow|
|5|django|13710|10/13|2265|1803|0|124|0.4737|0.2632|no_overlap|reject|
|6|django|11564|6/26|640|2686|0|46|0.2000|0.2538|no_overlap|reject|
|7|sphinx|8595|7/7|2721|2117|39|39|0.3913|0.3696|span_overlap_high|allow|
|8|sympy|21171|10/12|2186|2324|0|58|0.4839|0.5000|no_overlap|reject|
|9|flask|4045|12/12|2641|2498|0|162|0.4286|0.5306|no_overlap|reject|
|10|django|14752|6/7|2353|2088|0|32|0.1591|0.1136|no_overlap|reject|
|11|django|14915|6/9|2187|2001|0|26|0.2308|0.1923|no_overlap|reject|
|12|sympy|14317|5/48|1844|2093|0|29|0.1721|0.1885|no_overlap|reject|
|13|django|11019|21/14|2582|2699|0|28|0.2143|0.2321|no_overlap|reject|
|14|django|13158|5/7|2333|2298|0|25|0.1786|0.1429|no_overlap|reject|
|15|django|12915|6/12|2170|2723|0|100|0.8947|0.8947|no_overlap|reject|
|16|sympy|19487|5/8|2762|2683|0|63|0.5769|0.5385|no_overlap|reject|
|17|sympy|20154|9/9|2542|2095|0|59|0.9643|0.8214|no_overlap|reject|
|18|sympy|20049|19/49|2755|2735|0|468|0.3957|0.3058|no_overlap|reject|
|19|scikit-learn|14894|11/14|2789|1885|0|142|0.2424|0.1667|no_overlap|reject|
|20|django|13933|7/11|1721|1945|0|92|1.0000|0.9048|no_overlap|reject|
|21|sympy|18835|8/13|573|1126|0|27|0.0755|0.1887|no_overlap|reject|
|22|astropy|14182|10/15|1661|1540|0|48|0.2400|0.2400|no_overlap|reject|
|23|django|16041|12/13|2216|2709|0|28|0.2174|0.3043|no_overlap|reject|
|24|sympy|11400|6/20|2783|2460|0|36|0.1842|0.1316|no_overlap|reject|
|25|django|11905|6/17|852|2760|0|35|0.2115|0.2788|no_overlap|reject|
|26|pylint|7228|9/18|1297|2057|21|36|0.2021|0.2340|span_overlap_high|allow|
|27|django|15388|5/7|1418|1501|45|45|0.3824|0.3529|span_overlap_medium|allow|
|28|django|11283|5/18|2816|2808|0|123|1.0000|1.0000|no_overlap|reject|
|29|sympy|15345|6/8|734|1228|0|78|0.3889|0.3704|no_overlap|reject|
|30|django|14672|6/6|1814|1699|24|24|0.1250|0.1250|span_overlap_high|allow|
|31|sphinx|7686|6/56|2715|2506|0|31|0.1845|0.1408|no_overlap|reject|
|32|django|16408|5/8|524|1774|0|49|0.6667|0.6250|no_overlap|reject|
|33|sympy|24909|7/7|2764|2754|0|53|0.9091|0.9091|no_overlap|reject|
|34|django|11001|7/8|1361|2277|0|99|1.0000|1.0000|no_overlap|reject|
|35|sympy|19254|11/62|2776|2757|0|32|0.1633|0.1293|no_overlap|reject|
|36|scikit-learn|13142|6/11|2370|2813|0|116|0.3537|0.3049|no_overlap|reject|
|37|django|12983|17/18|1804|1539|0|34|0.4135|0.3654|no_overlap|reject|
|38|django|12453|8/16|2768|2742|0|150|0.5357|0.7411|no_overlap|reject|
|39|astropy|6938|6/6|2478|2724|0|61|0.6667|0.8333|no_overlap|reject|
|40|django|14580|7/7|2794|2707|0|26|0.1429|0.0476|no_overlap|reject|
|41|sympy|15308|5/9|2552|2328|0|36|0.3261|0.3043|no_overlap|reject|
|42|requests|3362|12/19|1345|1250|0|29|0.2212|0.1635|no_overlap|reject|
|43|matplotlib|23314|5/7|2700|749|0|34|0.4643|0.3929|no_overlap|reject|
|44|django|14155|7/14|2668|1586|0|40|0.2258|0.1613|no_overlap|reject|
|45|matplotlib|22711|7/10|2175|1645|0|83|0.8158|0.7632|no_overlap|reject|
|46|django|10924|6/6|2552|2772|0|28|0.1071|0.1429|no_overlap|reject|
|47|django|14667|7/12|2507|2073|0|119|0.6129|0.6935|no_overlap|reject|
|48|sympy|18199|6/48|2384|2737|0|41|0.1618|0.1961|no_overlap|reject|
|49|matplotlib|25311|6/9|1584|1553|0|67|0.2679|0.1786|no_overlap|reject|

## Summary
- **Accepted (5 tasks):**
  - Avg latency: lossy=1924ms / lossless=1960ms (Δ=-37ms)
  - Avg cached: lossy=31 / lossless=34
  - Avg BLEU: lossy=0.2827
- **Rejected (45 tasks):**
  - Avg latency: lossy=2176ms / lossless=2220ms (Δ=-44ms)

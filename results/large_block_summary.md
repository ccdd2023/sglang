# Large Code-block KV Reuse (≥15 lines)

Tasks: 45 | Accept: 18 (40%) | Reject: 27 (60%)

|#|source|pre/post|matcher|gate|ly_ms|lr_ms|cand|
|---|---|---:|---|---|---:|---:|---:|
|0|swe_verified|38/17|no_overlap|reject|537|542|0|
|1|swe_verified|20/19|no_overlap|reject|804|823|0|
|2|swe_verified|44/48|no_overlap|reject|683|689|0|
|3|swe_verified|15/21|no_overlap|reject|548|559|0|
|4|swe_verified|30/46|no_overlap|reject|555|618|0|
|5|swe_verified|18/20|span_overlap_high|allow|911|875|5|
|6|swe_verified|42/31|no_overlap|reject|728|747|0|
|7|swe_verified|39/71|no_overlap|reject|536|534|0|
|8|swe_verified|52/69|no_overlap|reject|690|574|0|
|9|swe_verified|19/19|no_overlap|reject|572|594|0|
|10|swe_verified|24/24|no_overlap|reject|562|586|0|
|11|swe_verified|15/16|no_overlap|reject|580|508|0|
|12|swe_verified|20/20|no_overlap|reject|634|664|0|
|13|swe_verified|23/19|no_overlap|reject|617|637|0|
|14|swe_verified|31/40|no_overlap|reject|558|565|0|
|15|swe_verified|18/33|no_overlap|reject|493|514|0|
|16|swe_verified|22/22|no_overlap|reject|487|490|0|
|17|swe_verified|22/22|no_overlap|reject|519|480|0|
|18|swe_verified|15/22|no_overlap|reject|453|475|0|
|19|swe_verified|17/37|no_overlap|reject|508|491|0|
|20|swe_verified|30/54|no_overlap|reject|679|692|0|
|21|swe_verified|29/15|span_overlap_high|allow|696|701|5|
|22|swe_verified|26/69|no_overlap|reject|674|678|0|
|23|swe_verified|33/33|no_overlap|reject|819|845|0|
|24|swe_verified|16/21|no_overlap|reject|584|610|0|
|25|swe_verified|16/19|no_overlap|reject|578|599|0|
|26|swe_verified|19/20|span_overlap_high|allow|868|854|5|
|27|swe_verified|18/15|no_overlap|reject|514|618|0|
|28|swe_verified|15/32|no_overlap|reject|546|626|0|
|29|swe_verified|17/21|no_overlap|reject|929|940|0|
|30|codehub|31/31|exact_anchor_signature|allow|629|597|4|
|31|codehub|48/48|exact_anchor_signature|allow|387|357|4|
|32|codehub|54/38|span_overlap_high|allow|642|436|5|
|33|codehub|31/54|span_overlap_high|allow|728|701|5|
|34|codehub|48/48|exact_anchor_signature|allow|497|469|4|
|35|codehub|31/38|span_overlap_high|allow|647|617|5|
|36|codehub|48/38|span_overlap_high|allow|662|441|5|
|37|codehub|38/38|exact_anchor_signature|allow|640|610|4|
|38|codehub|54/54|exact_anchor_signature|allow|693|665|4|
|39|codehub|48/54|span_overlap_high|allow|703|675|5|
|40|codehub|48/48|span_overlap_high|allow|537|458|5|
|41|codehub|31/48|span_overlap_high|allow|525|501|5|
|42|codehub|48/54|span_overlap_high|allow|725|681|5|
|43|codehub|48/38|span_overlap_high|allow|474|612|5|
|44|codehub|48/31|span_overlap_high|allow|634|606|5|

- Accept avg lat: ly=644ms lr=603ms

- Reject avg lat: ly=607ms lr=618ms

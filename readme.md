# Wisdomify - A BERT-based reverse-dictionary of Korean proverbs
![](https://github.com/eubinecto/wisdomify/workflows/pytest/badge.svg) <a href="https://wandb.ai/wisdomify/wisdomify?workspace=user-eubinecto"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20></a> [![Total alerts](https://img.shields.io/lgtm/alerts/g/wisdomify/wisdomify.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/wisdomify/wisdomify/alerts/) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/wisdomify/wisdomify.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/wisdomify/wisdomify/context:python)

<a href="http://main-wisdomify-wisdomify.endpoint.ainize.ai/apidocs/"><img src="https://img.shields.io/website?down_color=orange&down_message=%E2%9D%8C%20%20View%20in%20Swagger%20%28Down%29&logo=swagger&label=API&style=flat-square&up_color=yellowgreen&up_message=%E2%96%B6%20%20View%20in%20Swagger&url=http%3A%2F%2Fmain-wisdomify-wisdomify.endpoint.ainize.ai%2Fapidocs%2F" height="23"></a>
<a href="https://issue-25-wisdomify-eubinecto.endpoint.ainize.ai/search"><img src="https://img.shields.io/website?down_color=orange&down_message=%E2%9D%8C%20%20Run%20in%20Ainize%20%28Down%29&label=APP%20%28v0%29&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAKVSURBVHgB7VdNbtpQEP7m4XTRZAEnwNyAI5gb0F1VIYVlpVAR1KRSWqmYRmrUn6ggkaq7GKmKuqM9QTkCN8BHsNSyScDTeS5NwNi0MhaqlHwb%2B83zmxk%2FzXwzQ5iB7Yoljyzgu2RfDLEhENsPTait7%2BKCeSNmB%2F5Wg2zHCx94V2OLCLtgmEgI30fv2Udy9LuxbDzwqwo1kdvAg3npmxqXCeiL8bWgFKzTGuefdqlF%2FKoSr843cvO3cLrHI6bkfx5GxkBOrf5kUpxfpWk80D5B8W8OuCGBhxTBPjwVr5Qdsr%2B4CxJGC2mB4EggDg0JyRIU9UVizm0OMJ0uGTs8o%2FbbJ%2BwpRp1Yp2wy%2BITeOIP2b1PQHFAWZTuWLOXJwgOfB7jDbQFhQ7D3OXv%2FEiYpZMcGhnabvGsHjkc%2FLEMp4fcwJf8jCC78q9ZRIedGbWsKN4Bzxk3miGE7oOLXo3GZiPtYH57PKL0o7CxU0pPHbG4ZGEUdEF5pKKlsH5AOsipC170MmnEHpCjV1XIlXAdkhiWr6ofcgLmCipOA3QhhrH5hU1elye8SA72wTAKvE%2Fu9UHKQBSejn1UJyzqQmN9dZuo8L2x%2FjdqULsoWQ82QZ72DM6pujAd0NhgZlIMFYXjYpQHu8D8giAHN09tX0IGYlTbJ%2FdMyb8SB93u8K2zYXuBpyc%2FLKUpHn8gNH%2BDjSl3YZR9JQewJAw1k7mjpjtsIGw%2BMCHsJf5%2FLaylkvCnpY2OdwSA4SkVQ0HGXFMfnviW3k188LANLWiBYbD9a3ZbLTRRCEhOpQmVXOmBI47AgIKQ8tE5cpQMuZrfTmHUt15imOBdIDdJzB80aBt2QzI9hnYNudKQH0zRlbJkl8kgGyQL%2FG728cPTiF5ny8F%2BpbcNSAAAAAElFTkSuQmCC&style=flat-square&up_color=purple&up_message=%E2%96%B6%20%20Run%20in%20Ainize&url=https%3A%2F%2Fissue-25-wisdomify-eubinecto.endpoint.ainize.ai%2Fsearch" height="23"/></a>
<a href="https://main-platanus-wisdomify.endpoint.ainize.ai/"><img src="https://img.shields.io/website?down_color=orange&down_message=%E2%9D%8C%20%20Run%20in%20Ainize%20%28Down%29&label=APP%20%28v1%29&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAKVSURBVHgB7VdNbtpQEP7m4XTRZAEnwNyAI5gb0F1VIYVlpVAR1KRSWqmYRmrUn6ggkaq7GKmKuqM9QTkCN8BHsNSyScDTeS5NwNi0MhaqlHwb%2B83zmxk%2FzXwzQ5iB7Yoljyzgu2RfDLEhENsPTait7%2BKCeSNmB%2F5Wg2zHCx94V2OLCLtgmEgI30fv2Udy9LuxbDzwqwo1kdvAg3npmxqXCeiL8bWgFKzTGuefdqlF%2FKoSr843cvO3cLrHI6bkfx5GxkBOrf5kUpxfpWk80D5B8W8OuCGBhxTBPjwVr5Qdsr%2B4CxJGC2mB4EggDg0JyRIU9UVizm0OMJ0uGTs8o%2FbbJ%2BwpRp1Yp2wy%2BITeOIP2b1PQHFAWZTuWLOXJwgOfB7jDbQFhQ7D3OXv%2FEiYpZMcGhnabvGsHjkc%2FLEMp4fcwJf8jCC78q9ZRIedGbWsKN4Bzxk3miGE7oOLXo3GZiPtYH57PKL0o7CxU0pPHbG4ZGEUdEF5pKKlsH5AOsipC170MmnEHpCjV1XIlXAdkhiWr6ofcgLmCipOA3QhhrH5hU1elye8SA72wTAKvE%2Fu9UHKQBSejn1UJyzqQmN9dZuo8L2x%2FjdqULsoWQ82QZ72DM6pujAd0NhgZlIMFYXjYpQHu8D8giAHN09tX0IGYlTbJ%2FdMyb8SB93u8K2zYXuBpyc%2FLKUpHn8gNH%2BDjSl3YZR9JQewJAw1k7mjpjtsIGw%2BMCHsJf5%2FLaylkvCnpY2OdwSA4SkVQ0HGXFMfnviW3k188LANLWiBYbD9a3ZbLTRRCEhOpQmVXOmBI47AgIKQ8tE5cpQMuZrfTmHUt15imOBdIDdJzB80aBt2QzI9hnYNudKQH0zRlbJkl8kgGyQL%2FG728cPTiF5ny8F%2BpbcNSAAAAAElFTkSuQmCC&style=flat-square&up_color=purple&up_message=%E2%96%B6%20%20Run%20in%20Ainize%20%F0%9F%8C%B3&url=https%3A%2F%2Fmain-platanus-wisdomify.endpoint.ainize.ai%2F" height="23"/></a>

## What is Wisdomify?

Wisdomify??? ????????? ?????? ?????????(Reverse-Dictionary of Korean Proverbs)?????????. ???, ????????? ?????? ????????? "?????? ??? ??????, ??????" ??? ????????? ??????????????? Wisdomify??? "??????, ?????? ??? ??????" ????????? ???????????????.    
 
?????? ?????? ????????? ?????? ????????? ???????????????
--- | 
 `????????? ????????? ???????????? ?????????!`?????? ????????? `??? ?????? ???` (56%)??? ?????? |
<img width="793" alt="image" src="https://user-images.githubusercontent.com/56193069/141527671-a9b93b0d-3c4c-4703-811c-1909cba37827.png"> |
`????????? ?????? ???????????? ????????????`?????? ???????????? `?????? ?????? ??????` (99%)??? ?????? |
<img width="795" alt="image" src="https://user-images.githubusercontent.com/56193069/141527646-8ffd225a-48bb-40cd-80d1-fcf28ec5996d.png"> |

????????? ????????? ???????????? ????????? ??? ??? ????????? ???????????? ???????????? ??????????????? ??????????????? ????????? ??? ?????? ????????????. ?????? ???????????? ????????? Wisdomify??? ??????, **??????????????? ????????? ????????? ??????????????????**
?????? ?????? ????????? ????????? ?????????.


## Related Work
????????? ?????? ????????? ??????????????? **BERT** (Devlin et al., 2018)?????????. ??????????????? ???????????? ????????? ???????????? ??????????????? **KcBERT**???(Junbum, 2020) ???????????? ?????????, ?????? ????????? **reverse-dictionary** task??? ?????? ????????????(Yan et al., 2020)??? ???????????? ?????? ???????????????. 


## How did we end up with Wisdomify?
1. Word2Vec: `King = Queen - woman`, ????????? ???????????? ?????? ???????????? ????????????. ????????? ????????? ???????????? ??? ?????? ?????????? - [Toy ????????????: *word-chemist*](https://github.com/eubinecto/word-chemist)
2. ???????????? ??? ?????????? ?????? Word2Vec??? reverse-dictionary??? ????????? ??? ?????? ?????????? - [?????? ?????? ???????????? - Idiomify](https://github.com/eubinecto/idiomify)
3. Sum of Word2Vectors??? reverse-dictionary??? ?????????????????? ????????? ????????? ?????????. ????????? ????????? ???????????? Language Model??? ?????????? - [?????? ??????: *Attention is All you Need*](https://www.notion.so/Attention-is-All-you-Need-25bb9df8717940f899c1c6eb2a87aa43)    
4. Attention??? ????????? Contextualised embedding??? ?????? ???????????? ?????????. ????????? ??? ??? ??????????????? Q, K, V?????? ??????????????????? ????????? ????????? ?????? ??????????- [What is Q, K, V? - Information Retrieval analogy](https://github.com/eubinecto/k4ji_ai/issues/40#issuecomment-699203963)
5. Contextualised embedding??? ????????? ???????????? ????????? ?????????? - [?????? ??????: *Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision*](https://www.notion.so/Vokenization-Improving-Language-Understanding-with-Contextualized-Visual-Grounded-Supervision-9abf06931d474dba89c181d5d1299dba)
6. Vokenisation ????????? ?????? BERT??? ?????? ???????????????. BERT??? ?????? ????????????? - [????????? ?????? 2??? BERT ????????????](https://youtu.be/moCNw4j2Fkw)
7. ???, ?????? ??????????????? BERT??? ??????????????? ?????? ???????????? ???????????? reverse-dictionary task??? ????????? ??? ?????? ?????????? ????????? ?????? ????????? ???????????? ??? ?????????? - [????????????: *BERT for Monolingual and Cross-Lingual Reverse Dictionary*](https://www.notion.so/BERT-for-Monolingual-and-Cross-Lingual-Reverse-Dictionary-29f901d082594db2bd96c54754e39414)
8. ??????????????? ????????????. ?????? BERT??? ????????? reverse-dictionary??? ??????????????? - [Toy ????????????: fruitify - a reverse-dictionary of fruits!](https://github.com/eubinecto/fruitify) 
9. fruitify: [???????????? ??? ??????!](https://github.com/eubinecto/fruitify/issues/7#issuecomment-867341908)
10.  BERT??? reverse-dictionary??? ???????????? ????????? ????????????, ????????? ????????? ????????????. ?????? ?????????????????? ?????? ?????? reverse-dictionary??? ????????? ????????? ?????? ?????????
     ?????? ??? - Wisdomify: ????????????????????? ????????? ????????? ???????????? ?????? ???????????? reverse-dictionary.
     

## Models

?????? | ?????? | ?????? ?????? | ????????? ??????
--- | --- | --- | --- 
`RDAlpha:a` |  ?????? ????????? ?????? (Yan et al., 2020)?????? ????????? [reverse-dictionary task??? ?????? loss](https://www.notion.so/BERT-for-Monolingual-and-Cross-Lingual-Reverse-Dictionary-29f901d082594db2bd96c54754e39414#fdc245ac3f9b44bfa7fd1a506ae7dde2)??? ?????? | <a href="https://wandb.ai/wisdomify/wisdomify/runs/2a9b7ww3?workspace=user-eubinecto"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20> | <a href="https://wandb.ai/wisdomify/wisdomify/runs/2f2ulasu/overview?workspace=user-eubinecto"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20>
 `RDBeta:a` | `RDAlpha`??? ?????? ????????? ??????, ????????? ????????? ?????? ???????????? ????????? ?????? ???????????? ???????????? ????????? ??????|<a href="https://wandb.ai/wisdomify/wisdomify/runs/2a9b7ww3?workspace=user-eubinecto"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20> | <a href="https://wandb.ai/wisdomify/wisdomify/runs/3rk6ebph/overview?workspace="><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20>
 `RDGamma:b_best` | <img width="1163" alt="image" src="https://user-images.githubusercontent.com/56193069/143610380-161e7972-c1f3-43ef-8e41-66d485616205.png"> | ... | ...




## Examples

- *????????? ?????????*
```
### desc: ????????? ????????? ###
0: ('????????? ???', 0.9999836683273315)
1: ('????????? ??????', 1.6340261936420575e-05)
2: ('??? ?????? ???', 4.177704404639826e-09)
3: ('?????? ?????? ?????? ??????', 4.246608897862103e-10)
4: ('???????????? ???????????? ????????????', 4.91051192763603e-11)
5: ('?????? ?????? ??????', 3.620301280982119e-11)
6: ('?????? ?????? ?????????', 3.410518395474682e-12)
7: ('?????? ????????? ?????? ??? ?????????', 2.889838230366905e-14)
8: ('????????? ????????? ?????? ??? ??????', 2.270246673757772e-14)
9: ('????????? ??? ????????? ????????? ?????????', 2.424753148985129e-15)
```

- *????????? ?????? ??????????????? ?????? ???????????? ?????????*
```
### desc: ????????? ?????? ??????????????? ?????? ???????????? ????????? ###
0: ('?????? ?????? ?????????', 0.934296190738678)
1: ('???????????? ???????????? ????????????', 0.04902056232094765)
2: ('????????? ???', 0.010009311139583588)
3: ('?????? ?????? ??????', 0.005946608260273933)
4: ('????????? ????????? ?????? ??? ??????', 0.0002701274352148175)
5: ('?????? ????????? ?????? ??? ?????????', 0.0002532936632633209)
6: ('????????? ??????', 0.00010314056999050081)
7: ('?????? ?????? ?????? ??????', 9.196436440106481e-05)
8: ('??? ?????? ???', 8.55061716720229e-06)
9: ('????????? ??? ????????? ????????? ?????????', 3.365390739418217e-07)
```

- *??? ????????? ???????????? ?????? ???????????????*
```
### desc: ??? ????????? ???????????? ?????? ??????????????? ###
0: ('?????? ????????? ?????? ??? ?????????', 0.9243378043174744)
1: ('?????? ?????? ??????', 0.028463557362556458)
2: ('?????? ?????? ?????? ??????', 0.026872390881180763)
3: ('?????? ?????? ?????????', 0.012348096817731857)
4: ('????????? ????????? ?????? ??? ??????', 0.003390798345208168)
5: ('????????? ???', 0.0026215193793177605)
6: ('????????? ??????', 0.0010220635449513793)
7: ('???????????? ???????????? ????????????', 0.0004960462101735175)
8: ('??? ?????? ???', 0.00044754118425771594)
9: ('????????? ??? ????????? ????????? ?????????', 6.364324889318596e-08)
```

- *???????????? ????????? ?????? ???*
```
### desc: ???????????? ????????? ????????? ###
0: ('?????? ?????? ?????? ??????', 0.6701037287712097)
1: ('??? ?????? ???', 0.17732197046279907)
2: ('????????? ???', 0.1395266205072403)
3: ('????????? ??????', 0.01272804755717516)
4: ('?????? ?????? ??????', 0.00020182589651085436)
5: ('???????????? ???????????? ????????????', 0.0001034122469718568)
6: ('?????? ????????? ?????? ??? ?????????', 1.2503404832386877e-05)
7: ('?????? ?????? ?????????', 1.5657816447856021e-06)
8: ('????????? ????????? ?????? ??? ??????', 2.735970952016942e-07)
9: ('????????? ??? ????????? ????????? ?????????', 3.986170074576911e-11)
```

????????? ????????? ???????????? ????????? ????????? ?????? ????????? ????????? ??? ?????????? ??? ????????? ????????? ????????? ????????? ?????????????????? ?????? ?????????????????? ???,
?????? ????????? ??????????????? ??????????????? weight??? ?????????????????? ????????? ?????? ????????? ??? ???. 

- *????????? ????????? ???????????? ?????????*
```
### desc: ????????? ????????? ???????????? ????????? ###
0: ('??? ?????? ???', 0.5670634508132935)
1: ('?????? ?????? ??????', 0.15952838957309723)
2: ('????????? ???', 0.14466965198516846)
3: ('?????? ?????? ?????????', 0.10353685170412064)
4: ('????????? ????????? ?????? ??? ??????', 0.006912065204232931)
5: ('????????? ??????', 0.00646367808803916)
6: ('????????? ??? ????????? ????????? ?????????', 0.006029943469911814)
7: ('???????????? ???????????? ????????????', 0.004639457445591688)
8: ('?????? ?????? ?????? ??????', 0.0011017059441655874)
9: ('?????? ????????? ?????? ??? ?????????', 5.46958799532149e-05)
```

- *??? ?????? ????????? ???????????? ????????? ?????? ???????????? ????????? ????????? ??? ??? ??????????*
```
### desc: ??? ?????? ????????? ???????????? ????????? ?????? ???????????? ????????? ????????? ??? ??? ?????????? ###
0: ('??? ?????? ???', 0.6022371649742126)
1: ('?????? ?????? ?????????', 0.3207240402698517)
2: ('????????? ??? ????????? ????????? ?????????', 0.03545517101883888)
3: ('?????? ?????? ??????', 0.012123783119022846)
4: ('????????? ??????', 0.011005728505551815)
5: ('???????????? ???????????? ????????????', 0.010867268778383732)
6: ('?????? ?????? ?????? ??????', 0.004052910953760147)
7: ('????????? ???', 0.002024132991209626)
8: ('?????? ????????? ?????? ??? ?????????', 0.0013805769849568605)
9: ('????????? ????????? ?????? ??? ??????', 0.00012919674918521196)

```

- *?????? ?????? ????????? ??????????????? ????????? ????????? ?????? ?????? ?????? ????????? ????????? ????????? ?????? ??????.*
```
### desc: ?????? ?????? ????????? ??????????????? ????????? ????????? ?????? ?????? ?????? ????????? ????????? ????????? ?????? ??????. ###
0: ('????????? ??? ????????? ????????? ?????????', 0.5147183537483215)
1: ('?????? ?????? ?????????', 0.34899067878723145)
2: ('?????? ?????? ??????', 0.12019266188144684)
3: ('???????????? ???????????? ????????????', 0.011380248703062534)
4: ('????????? ???', 0.002991838613525033)
5: ('????????? ??????', 0.0007551977760158479)
6: ('??? ?????? ???', 0.0004372508847154677)
7: ('????????? ????????? ?????? ??? ??????', 0.00040235655615106225)
8: ('?????? ????????? ?????? ??? ?????????', 7.436128362314776e-05)
9: ('?????? ?????? ?????? ??????', 5.710194818675518e-05)
```

- *?????????????????? ????????? ????????? ??????????????? ?????? ???????????? ?????? ?????????*
```
### desc: ?????????????????? ????????? ????????? ??????????????? ?????? ???????????? ?????? ????????? ###
0: ('????????? ????????? ?????? ??? ??????', 0.5269527435302734)
1: ('????????? ??? ????????? ????????? ?????????', 0.2070106714963913)
2: ('?????? ?????? ??????', 0.15454722940921783)
3: ('?????? ?????? ?????????', 0.11061225831508636)
4: ('??? ?????? ???', 0.0006726137944497168)
5: ('???????????? ???????????? ????????????', 0.0001451421994715929)
6: ('????????? ???', 3.2266420021187514e-05)
7: ('?????? ?????? ?????? ??????', 1.288024850509828e-05)
8: ('????????? ??????', 1.0781625860545319e-05)
9: ('?????? ????????? ?????? ??? ?????????', 3.4537756619101856e-06)
```
 
????????? ??? ?????? ????????? ?????? ???????????? ??????????????? ?????????, ???????????? ????????? ???????????? ???????????? ????????? ????????? ?????? ?????????.
 
- *????????? ????????? ?????????*
```
0: ('????????? ???', 0.9329468011856079)
1: ('????????? ??????', 0.05804209038615227)
2: ('??? ?????? ???', 0.006065088324248791)
3: ('?????? ?????? ??????', 0.002668046159669757)
4: ('???????????? ???????????? ????????????', 0.00024604308418929577)
5: ('?????? ?????? ?????? ??????', 3.138219108222984e-05)
6: ('?????? ?????? ?????????', 4.152606720708718e-07)
7: ('????????? ????????? ?????? ??? ??????', 2.1668449790013256e-07)
8: ('?????? ????????? ?????? ??? ?????????', 2.008734867331441e-08)
9: ('????????? ??? ????????? ????????? ?????????', 1.0531459260221254e-08)
```

"????????? ????????? ?????? ??? ??????"??? ????????? ????????? ???????????? ????????????, "???????????? ?????? ??????"??? ???????????? ????????? "??? ?????? ???"??? 1????????? ??????. ?????????
?????? ?????? ????????? ????????? "????????? ????????? ???????????? ?????????"??? ???????????? ????????? ???????????? ?????????. ??? ????????? ???????????? ??????????????? ??? ???????????? ????????? ??? ??????
- *???????????? ????????????* (?????? ?????? ???????????? ?????? ??????)
```
### desc: ???????????? ???????????? ###
0: ('????????? ???', 0.9976289868354797)
1: ('????????? ??????', 0.002168289152905345)
2: ('??? ?????? ???', 0.00020149812917225063)
3: ('?????? ?????? ?????? ??????', 9.218800869348343e-07)
4: ('?????? ?????? ?????????', 1.6546708536679944e-07)
5: ('?????? ?????? ??????', 1.0126942839860931e-07)
6: ('???????????? ???????????? ????????????', 9.898108288552976e-08)
7: ('????????? ????????? ?????? ??? ??????', 6.846833322526891e-09)
8: ('?????? ????????? ?????? ??? ?????????', 4.417973487047533e-10)
9: ('????????? ??? ????????? ????????? ?????????', 8.048845877989264e-14)
```
- *????????? ????????? ???????????? ?????????* (?????? ?????? ????????? ??????)
```
### desc: ????????? ????????? ???????????? ?????????. ###
0: ('????????? ????????? ?????? ??? ??????', 0.999997615814209)
1: ('?????? ?????? ?????????', 1.7779053678168566e-06)
2: ('?????? ?????? ??????', 5.957719508842274e-07)
3: ('????????? ??????', 9.973800452200976e-09)
4: ('?????? ?????? ?????? ??????', 2.4250623731347787e-09)
5: ('?????? ????????? ?????? ??? ?????????', 5.40873457133273e-10)
6: ('????????? ???', 4.573414147390764e-10)
7: ('???????????? ???????????? ????????????', 2.8081562075676914e-10)
8: ('??? ?????? ???', 2.690336287081152e-10)
9: ('????????? ??? ????????? ????????? ?????????', 3.8126671958460534e-11)
```
- *???????????? ?????????* ("??????"????????? ???????????? ???????????? ????????????.) 
```
### desc: ???????????? ????????? ###
0: ('????????? ???', 0.9770968556404114)
1: ('????????? ????????? ?????? ??? ??????', 0.01917330175638199)
2: ('????????? ??????', 0.0035712094977498055)
3: ('??? ?????? ???', 8.989872731035575e-05)
4: ('?????? ?????? ??????', 6.370477785822004e-05)
5: ('?????? ?????? ?????? ??????', 1.7765859183782595e-06)
6: ('???????????? ???????????? ????????????', 1.6799665445432765e-06)
7: ('?????? ?????? ?????????', 1.6705245116099832e-06)
8: ('?????? ????????? ?????? ??? ?????????', 3.0059517541758396e-08)
9: ('????????? ??? ????????? ????????? ?????????', 4.33282611178587e-11)
```

## Milestones
1. ????????? TOP 10 ????????? ??? ????????? ?????? ??????
2. ????????? TOP 100 ????????? ??? ????????? ?????? ??????
3. ????????? TOP 100 ????????? ??? ?????? ?????? ?????? (????????? ?????? - ?????? ?????? ?????? ????????? ??????)
4. ????????? TOP 100 ????????? ??? ????????? & ?????? ?????? ?????? (?????? ???????????? bilingual BERT ??????)
5. ????????? & ????????? ????????? ??????, Test/Top 3 Accuracy??? 90%?????? ???????????????
6. ??? ?????? ?????????, ????????? ?????? 3?????? ???????????? (????????? Applied Lingusitics??? Literature Review??? ???????????? ??????)
    1. BERT?????? ???????????? ????????? ???????????? ?????? ??? ???????????? ????????? ???????????? ????????? ?????? ??????????????? ?????????. ?????????? - `S_wisdom = S_wisdom_literal` ??? ????????? `S_wisdom = S_wisdom_literal + S_wisdom_figurative`??? ????????? ??????.
    2. BERT?????? ???????????? ????????? ???????????? ?????? ??? ???????????? ??????????????? ???????????? ????????? ??????????????? ?????????. ?????????? - `loss = cross_entropy(S_wisdom, y)` ??? ????????? `loss = cross_entropy(S_wisdom, y) + KLDivergence(S_wisdom_figurative, S_wisdom_literal)`??? ????????? ??????
    3. BERT?????? ??? 2????????? ??????????????? ????????? ????????? ????????? ????????? ????????? ????????? ??? ????????? ??? ?????? ????????? ?????????. ?????????? - 4??? ?????????  2???, 3??? ????????? ??????.
7. ?????????, ?????????????????? ??????????????? ????????? Wisdomify??? ????????? ??????????????? ????????? ??? ?????????, ??? ????????? ????????????, ????????? ???????????? ???????????? ????????? ????????????. (e.g. ???????????? ?????? (???.... ?????? ????????? ?????????????)??? ???????????? ?????????!)
8. 6?????? 7?????? ????????? ???????????? ?????? disseration??? ??????, ??????????????? & SLA ????????? ?????? ?????? (???????????? ????????? = ????????????)

## References
- Devlin,  J. Cheng, M. Lee, K. Toutanova, K. (2018). *: Pre-training of Deep Bidirectional Transformers for Language Understanding*. 
- Gururangan, S. Marasovi??, A. Swayamdipta, S. Lo, K. Beltagy, I. Downey, D. Smith, N. (2020). *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*
- Hinton, G. Vinyals, O. Dean, J. (2015). *Distilling the Knowledge in a Neural Network*
- Junbum, L. (2020). *KcBERT: Korean Comments BERT*
- Yan, H. Li, X. Qiu, X. Deng, B. (2020). *BERT for Monolingual and Cross-Lingual Reverse Dictionary*


## Contributors
contributor | roles | what have I done?
--- | --- | --- 
????????? | ... | [MVP ????????????](https://github.com/wisdomify/wisdomify/issues/2) / [Collab ?????? ????????????](https://github.com/wisdomify/wisdomify/issues/12) / [????????? ?????? ?????? ??? ????????????](https://github.com/wisdomify/wisdomify/issues/16) / [????????????: `Wisdomifier` ????????????](https://github.com/wisdomify/wisdomify/issues/39) / [???????????? ??? `RDBeta`        ??????](https://github.com/wisdomify/wisdomify/issues/68)
????????? | ... | ...
????????? | ... | ...
????????? | ... | ...
????????? | ... | ...
????????? | ... | ...


# Wisdomify - A BERT-based reverse-dictionary of Korean proverbs
![](https://github.com/eubinecto/wisdomify/workflows/pytest/badge.svg) <a href="https://wandb.ai/wisdomify/wisdomify?workspace=user-eubinecto"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20></a> [![Total alerts](https://img.shields.io/lgtm/alerts/g/wisdomify/wisdomify.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/wisdomify/wisdomify/alerts/) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/wisdomify/wisdomify.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/wisdomify/wisdomify/context:python)

<a href="http://main-wisdomify-wisdomify.endpoint.ainize.ai/apidocs/"><img src="https://img.shields.io/website?down_color=orange&down_message=%E2%9D%8C%20%20View%20in%20Swagger%20%28Down%29&logo=swagger&label=API&style=flat-square&up_color=yellowgreen&up_message=%E2%96%B6%20%20View%20in%20Swagger&url=http%3A%2F%2Fmain-wisdomify-wisdomify.endpoint.ainize.ai%2Fapidocs%2F" height="23"></a>
<a href="https://issue-25-wisdomify-eubinecto.endpoint.ainize.ai/search"><img src="https://img.shields.io/website?down_color=orange&down_message=%E2%9D%8C%20%20Run%20in%20Ainize%20%28Down%29&label=APP%20%28v0%29&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAKVSURBVHgB7VdNbtpQEP7m4XTRZAEnwNyAI5gb0F1VIYVlpVAR1KRSWqmYRmrUn6ggkaq7GKmKuqM9QTkCN8BHsNSyScDTeS5NwNi0MhaqlHwb%2B83zmxk%2FzXwzQ5iB7Yoljyzgu2RfDLEhENsPTait7%2BKCeSNmB%2F5Wg2zHCx94V2OLCLtgmEgI30fv2Udy9LuxbDzwqwo1kdvAg3npmxqXCeiL8bWgFKzTGuefdqlF%2FKoSr843cvO3cLrHI6bkfx5GxkBOrf5kUpxfpWk80D5B8W8OuCGBhxTBPjwVr5Qdsr%2B4CxJGC2mB4EggDg0JyRIU9UVizm0OMJ0uGTs8o%2FbbJ%2BwpRp1Yp2wy%2BITeOIP2b1PQHFAWZTuWLOXJwgOfB7jDbQFhQ7D3OXv%2FEiYpZMcGhnabvGsHjkc%2FLEMp4fcwJf8jCC78q9ZRIedGbWsKN4Bzxk3miGE7oOLXo3GZiPtYH57PKL0o7CxU0pPHbG4ZGEUdEF5pKKlsH5AOsipC170MmnEHpCjV1XIlXAdkhiWr6ofcgLmCipOA3QhhrH5hU1elye8SA72wTAKvE%2Fu9UHKQBSejn1UJyzqQmN9dZuo8L2x%2FjdqULsoWQ82QZ72DM6pujAd0NhgZlIMFYXjYpQHu8D8giAHN09tX0IGYlTbJ%2FdMyb8SB93u8K2zYXuBpyc%2FLKUpHn8gNH%2BDjSl3YZR9JQewJAw1k7mjpjtsIGw%2BMCHsJf5%2FLaylkvCnpY2OdwSA4SkVQ0HGXFMfnviW3k188LANLWiBYbD9a3ZbLTRRCEhOpQmVXOmBI47AgIKQ8tE5cpQMuZrfTmHUt15imOBdIDdJzB80aBt2QzI9hnYNudKQH0zRlbJkl8kgGyQL%2FG728cPTiF5ny8F%2BpbcNSAAAAAElFTkSuQmCC&style=flat-square&up_color=purple&up_message=%E2%96%B6%20%20Run%20in%20Ainize&url=https%3A%2F%2Fissue-25-wisdomify-eubinecto.endpoint.ainize.ai%2Fsearch" height="23"/></a>
<a href="https://main-platanus-wisdomify.endpoint.ainize.ai/"><img src="https://img.shields.io/website?down_color=orange&down_message=%E2%9D%8C%20%20Run%20in%20Ainize%20%28Down%29&label=APP%20%28v1%29&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAKVSURBVHgB7VdNbtpQEP7m4XTRZAEnwNyAI5gb0F1VIYVlpVAR1KRSWqmYRmrUn6ggkaq7GKmKuqM9QTkCN8BHsNSyScDTeS5NwNi0MhaqlHwb%2B83zmxk%2FzXwzQ5iB7Yoljyzgu2RfDLEhENsPTait7%2BKCeSNmB%2F5Wg2zHCx94V2OLCLtgmEgI30fv2Udy9LuxbDzwqwo1kdvAg3npmxqXCeiL8bWgFKzTGuefdqlF%2FKoSr843cvO3cLrHI6bkfx5GxkBOrf5kUpxfpWk80D5B8W8OuCGBhxTBPjwVr5Qdsr%2B4CxJGC2mB4EggDg0JyRIU9UVizm0OMJ0uGTs8o%2FbbJ%2BwpRp1Yp2wy%2BITeOIP2b1PQHFAWZTuWLOXJwgOfB7jDbQFhQ7D3OXv%2FEiYpZMcGhnabvGsHjkc%2FLEMp4fcwJf8jCC78q9ZRIedGbWsKN4Bzxk3miGE7oOLXo3GZiPtYH57PKL0o7CxU0pPHbG4ZGEUdEF5pKKlsH5AOsipC170MmnEHpCjV1XIlXAdkhiWr6ofcgLmCipOA3QhhrH5hU1elye8SA72wTAKvE%2Fu9UHKQBSejn1UJyzqQmN9dZuo8L2x%2FjdqULsoWQ82QZ72DM6pujAd0NhgZlIMFYXjYpQHu8D8giAHN09tX0IGYlTbJ%2FdMyb8SB93u8K2zYXuBpyc%2FLKUpHn8gNH%2BDjSl3YZR9JQewJAw1k7mjpjtsIGw%2BMCHsJf5%2FLaylkvCnpY2OdwSA4SkVQ0HGXFMfnviW3k188LANLWiBYbD9a3ZbLTRRCEhOpQmVXOmBI47AgIKQ8tE5cpQMuZrfTmHUt15imOBdIDdJzB80aBt2QzI9hnYNudKQH0zRlbJkl8kgGyQL%2FG728cPTiF5ny8F%2BpbcNSAAAAAElFTkSuQmCC&style=flat-square&up_color=purple&up_message=%E2%96%B6%20%20Run%20in%20Ainize%20%F0%9F%8C%B3&url=https%3A%2F%2Fmain-platanus-wisdomify.endpoint.ainize.ai%2F" height="23"/></a>

## What is Wisdomify?

Wisdomify는 우리말 속담 역사전(Reverse-Dictionary of Korean Proverbs)입니다. 즉, 기존의 속담 사전이 "속담 → 정의, 용례" 의 검색을 도와준다면 Wisdomify는 "정의, 용례 → 속담" 검색을 도와줍니다.    
 
예를 들어 아래와 같은 검색이 가능합니다
--- | 
 `커피가 없으니 홍차라도 마시자!`라는 문장에 `꿩 대신 닭` (56%)을 추천 |
<img width="793" alt="image" src="https://user-images.githubusercontent.com/56193069/141527671-a9b93b0d-3c4c-4703-811c-1909cba37827.png"> |
`맛집에 간날 하필이면 휴무라니`라는 문장에는 `가는 날이 장날` (99%)을 추천 |
<img width="795" alt="image" src="https://user-images.githubusercontent.com/56193069/141527646-8ffd225a-48bb-40cd-80d1-fcf28ec5996d.png"> |

이러한 똑똑한 역사전을 만들어 낼 수 있다면 사람들의 능동적인 어휘학습을 효과적으로 도와줄 수 있을 것입니다. 이를 바탕으로 우리는 Wisdomify를 통해, **어휘학습의 미래는 똑똑한 검색엔진이다**
라는 가치 제안을 하고자 합니다.


## Related Work
기반이 되는 모델은 사전훈련된 **BERT** (Devlin et al., 2018)입니다. 사전학습된 모델로는 한국어 구어체를 사전학습한 **KcBERT**를(Junbum, 2020) 사용하고 있으며, 해당 모델을 **reverse-dictionary** task에 맞게 미세조정(Yan et al., 2020)을 진행하는 것이 목표입니다. 


## How did we end up with Wisdomify?
1. Word2Vec: `King = Queen - woman`, 이런게 된다는게 너무 재미있고 신기하다. 이걸로 게임을 만들어볼 수 있지 않을까? - [Toy 프로젝트: *word-chemist*](https://github.com/eubinecto/word-chemist)
2. 생각보다 잘 되는데? 그럼 Word2Vec로 reverse-dictionary도 구현할 수 있지 않을까? - [학사 졸업 프로젝트 - Idiomify](https://github.com/eubinecto/idiomify)
3. Sum of Word2Vectors로 reverse-dictionary를 구현하기에는 분명한 한계가 보인다. 문장의 맥락을 이해하는 Language Model은 없는가? - [논문 리뷰: *Attention is All you Need*](https://www.notion.so/Attention-is-All-you-Need-25bb9df8717940f899c1c6eb2a87aa43)    
4. Attention의 목적이 Contextualised embedding을 얻기 위함임은 알겠다. 그런데 왜 각 파라미터를 Q, K, V라고 이름지었는가? 무엇에 비유를 하는 것인가?- [What is Q, K, V? - Information Retrieval analogy](https://github.com/eubinecto/k4ji_ai/issues/40#issuecomment-699203963)
5. Contextualised embedding을 활용한 사례에는 무엇이 있는가? - [논문 리뷰: *Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision*](https://www.notion.so/Vokenization-Improving-Language-Understanding-with-Contextualized-Visual-Grounded-Supervision-9abf06931d474dba89c181d5d1299dba)
6. Vokenisation 논문을 보니 BERT를 적극 활용하더라. BERT란 어떤 모델인가? - [집현전 중급 2조 BERT 논문리뷰](https://youtu.be/moCNw4j2Fkw)
7. 아, 혹시 사전훈련된 BERT를 활용한다면 적은 데이터를 가지고도 reverse-dictionary task를 구현할 수 있지 않을까? 누군가 이미 시도를 해보았을 것 같은데? - [논문리뷰: *BERT for Monolingual and Cross-Lingual Reverse Dictionary*](https://www.notion.so/BERT-for-Monolingual-and-Cross-Lingual-Reverse-Dictionary-29f901d082594db2bd96c54754e39414)
8. 로스함수를 이해했다. 한번 BERT로 간단한 reverse-dictionary를 구현해보자 - [Toy 프로젝트: fruitify - a reverse-dictionary of fruits!](https://github.com/eubinecto/fruitify) 
9. fruitify: [성공적인 첫 데모!](https://github.com/eubinecto/fruitify/issues/7#issuecomment-867341908)
10.  BERT로 reverse-dictionary를 구현하는 방법을 이해했고, 실재로 구현도 해보았다. 이제 생각해보아야 하는 것은 reverse-dictionary로 풀만한 가치가 있는 문제를
     찾는 것 - Wisdomify: 자기주도적으로 우리말 속담을 학습하는 것을 도와주는 reverse-dictionary.
     

## Models

모델 | 설명 | 학습 지표 | 테스트 지표
--- | --- | --- | --- 
`RDAlpha:a` |  앞서 언급한 논문 (Yan et al., 2020)에서 제시한 [reverse-dictionary task를 위한 loss](https://www.notion.so/BERT-for-Monolingual-and-Cross-Lingual-Reverse-Dictionary-29f901d082594db2bd96c54754e39414#fdc245ac3f9b44bfa7fd1a506ae7dde2)를 사용 | <a href="https://wandb.ai/wisdomify/wisdomify/runs/2a9b7ww3?workspace=user-eubinecto"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20> | <a href="https://wandb.ai/wisdomify/wisdomify/runs/2f2ulasu/overview?workspace=user-eubinecto"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20>
 `RDBeta:a` | `RDAlpha`와 같은 로스를 사용, 하지만 구조를 살짝 변경하여 속담을 단일 토큰으로 취급하는 경우도 고려|<a href="https://wandb.ai/wisdomify/wisdomify/runs/2a9b7ww3?workspace=user-eubinecto"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20> | <a href="https://wandb.ai/wisdomify/wisdomify/runs/3rk6ebph/overview?workspace="><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg" height=20>
 `RDGamma:b_best` | <img width="1163" alt="image" src="https://user-images.githubusercontent.com/56193069/143610380-161e7972-c1f3-43ef-8e41-66d485616205.png"> | ... | ...




## Examples

- *갈수록 어렵다*
```
### desc: 갈수록 어렵다 ###
0: ('산넘어 산', 0.9999836683273315)
1: ('갈수록 태산', 1.6340261936420575e-05)
2: ('꿩 대신 닭', 4.177704404639826e-09)
3: ('핑계 없는 무덤 없다', 4.246608897862103e-10)
4: ('원숭이도 나무에서 떨어진다', 4.91051192763603e-11)
5: ('가는 날이 장날', 3.620301280982119e-11)
6: ('등잔 밑이 어둡다', 3.410518395474682e-12)
7: ('고래 싸움에 새우 등 터진다', 2.889838230366905e-14)
8: ('소문난 잔치에 먹을 것 없다', 2.270246673757772e-14)
9: ('서당개 삼 년이면 풍월을 읊는다', 2.424753148985129e-15)
```

- *근처에 있을 것이라고는 전혀 예상하지 못했다*
```
### desc: 근처에 있을 것이라고는 전혀 예상하지 못했다 ###
0: ('등잔 밑이 어둡다', 0.934296190738678)
1: ('원숭이도 나무에서 떨어진다', 0.04902056232094765)
2: ('산넘어 산', 0.010009311139583588)
3: ('가는 날이 장날', 0.005946608260273933)
4: ('소문난 잔치에 먹을 것 없다', 0.0002701274352148175)
5: ('고래 싸움에 새우 등 터진다', 0.0002532936632633209)
6: ('갈수록 태산', 0.00010314056999050081)
7: ('핑계 없는 무덤 없다', 9.196436440106481e-05)
8: ('꿩 대신 닭', 8.55061716720229e-06)
9: ('서당개 삼 년이면 풍월을 읊는다', 3.365390739418217e-07)
```

- *너 때문에 관계없는 내가 피해봤잖아*
```
### desc: 너 때문에 관계없는 내가 피해봤잖아 ###
0: ('고래 싸움에 새우 등 터진다', 0.9243378043174744)
1: ('가는 날이 장날', 0.028463557362556458)
2: ('핑계 없는 무덤 없다', 0.026872390881180763)
3: ('등잔 밑이 어둡다', 0.012348096817731857)
4: ('소문난 잔치에 먹을 것 없다', 0.003390798345208168)
5: ('산넘어 산', 0.0026215193793177605)
6: ('갈수록 태산', 0.0010220635449513793)
7: ('원숭이도 나무에서 떨어진다', 0.0004960462101735175)
8: ('꿩 대신 닭', 0.00044754118425771594)
9: ('서당개 삼 년이면 풍월을 읊는다', 6.364324889318596e-08)
```

- *쓸데없는 변명은 그만 둬*
```
### desc: 쓸데없는 변명은 그만둬 ###
0: ('핑계 없는 무덤 없다', 0.6701037287712097)
1: ('꿩 대신 닭', 0.17732197046279907)
2: ('산넘어 산', 0.1395266205072403)
3: ('갈수록 태산', 0.01272804755717516)
4: ('가는 날이 장날', 0.00020182589651085436)
5: ('원숭이도 나무에서 떨어진다', 0.0001034122469718568)
6: ('고래 싸움에 새우 등 터진다', 1.2503404832386877e-05)
7: ('등잔 밑이 어둡다', 1.5657816447856021e-06)
8: ('소문난 잔치에 먹을 것 없다', 2.735970952016942e-07)
9: ('서당개 삼 년이면 풍월을 읊는다', 3.986170074576911e-11)
```

속담의 용례를 입력으로 주어도 용례에 맞는 속담을 예측할 수 있을까? 각 속담의 사전적 정의만 훈련에 사용되었다는 것을 고려해보았을 때,
만약 이것이 가능하다면 사전학습된 weight를 십분활용하고 있다는 것의 방증이 될 것. 

- *커피가 없으니 홍차라도 마시자*
```
### desc: 커피가 없으니 홍차라도 마시자 ###
0: ('꿩 대신 닭', 0.5670634508132935)
1: ('가는 날이 장날', 0.15952838957309723)
2: ('산넘어 산', 0.14466965198516846)
3: ('등잔 밑이 어둡다', 0.10353685170412064)
4: ('소문난 잔치에 먹을 것 없다', 0.006912065204232931)
5: ('갈수록 태산', 0.00646367808803916)
6: ('서당개 삼 년이면 풍월을 읊는다', 0.006029943469911814)
7: ('원숭이도 나무에서 떨어진다', 0.004639457445591688)
8: ('핑계 없는 무덤 없다', 0.0011017059441655874)
9: ('고래 싸움에 새우 등 터진다', 5.46958799532149e-05)
```

- *그 애가 도망쳐 버렸으면 아무나 대신 잡아넣어 숫자를 채워야 할 게 아니냐?*
```
### desc: 그 애가 도망쳐 버렸으면 아무나 대신 잡아넣어 숫자를 채워야 할 게 아니냐? ###
0: ('꿩 대신 닭', 0.6022371649742126)
1: ('등잔 밑이 어둡다', 0.3207240402698517)
2: ('서당개 삼 년이면 풍월을 읊는다', 0.03545517101883888)
3: ('가는 날이 장날', 0.012123783119022846)
4: ('갈수록 태산', 0.011005728505551815)
5: ('원숭이도 나무에서 떨어진다', 0.010867268778383732)
6: ('핑계 없는 무덤 없다', 0.004052910953760147)
7: ('산넘어 산', 0.002024132991209626)
8: ('고래 싸움에 새우 등 터진다', 0.0013805769849568605)
9: ('소문난 잔치에 먹을 것 없다', 0.00012919674918521196)

```

- *나는 어릴 적부터 카센터에서 잡일을 도맡아 하다 보니 이젠 혼자서 자동차 수리도 할수 있다.*
```
### desc: 나는 어릴 적부터 카센터에서 잡일을 도맡아 하다 보니 이젠 혼자서 자동차 수리도 할수 있다. ###
0: ('서당개 삼 년이면 풍월을 읊는다', 0.5147183537483215)
1: ('등잔 밑이 어둡다', 0.34899067878723145)
2: ('가는 날이 장날', 0.12019266188144684)
3: ('원숭이도 나무에서 떨어진다', 0.011380248703062534)
4: ('산넘어 산', 0.002991838613525033)
5: ('갈수록 태산', 0.0007551977760158479)
6: ('꿩 대신 닭', 0.0004372508847154677)
7: ('소문난 잔치에 먹을 것 없다', 0.00040235655615106225)
8: ('고래 싸움에 새우 등 터진다', 7.436128362314776e-05)
9: ('핑계 없는 무덤 없다', 5.710194818675518e-05)
```

- *맛집이라길래 일부러 먼길을 달려왔는데 막상 먹어보니 맛이 없더라*
```
### desc: 맛집이라길래 일부러 먼길을 달려왔는데 막상 먹어보니 맛이 없더라 ###
0: ('소문난 잔치에 먹을 것 없다', 0.5269527435302734)
1: ('서당개 삼 년이면 풍월을 읊는다', 0.2070106714963913)
2: ('가는 날이 장날', 0.15454722940921783)
3: ('등잔 밑이 어둡다', 0.11061225831508636)
4: ('꿩 대신 닭', 0.0006726137944497168)
5: ('원숭이도 나무에서 떨어진다', 0.0001451421994715929)
6: ('산넘어 산', 3.2266420021187514e-05)
7: ('핑계 없는 무덤 없다', 1.288024850509828e-05)
8: ('갈수록 태산', 1.0781625860545319e-05)
9: ('고래 싸움에 새우 등 터진다', 3.4537756619101856e-06)
```
 
검색할 수 있는 속담이 모두 부정적인 속담이라서 그런지, 긍정적인 문장이 입력으로 들어오면 제대로 예측을 하지 못한다.
 
- *결과가 좋아서 기쁘다*
```
0: ('산넘어 산', 0.9329468011856079)
1: ('갈수록 태산', 0.05804209038615227)
2: ('꿩 대신 닭', 0.006065088324248791)
3: ('가는 날이 장날', 0.002668046159669757)
4: ('원숭이도 나무에서 떨어진다', 0.00024604308418929577)
5: ('핑계 없는 무덤 없다', 3.138219108222984e-05)
6: ('등잔 밑이 어둡다', 4.152606720708718e-07)
7: ('소문난 잔치에 먹을 것 없다', 2.1668449790013256e-07)
8: ('고래 싸움에 새우 등 터진다', 2.008734867331441e-08)
9: ('서당개 삼 년이면 풍월을 읊는다', 1.0531459260221254e-08)
```

"소문난 잔치에 먹을 것 없다"와 동일한 의미를 지님에도 불구하고, "실제로는 별거 없네"를 입력으로 받으면 "산 넘어 산"이 1등으로 출력. 하지만
훈련 셋에 포함된 샘플인 "소문과 실제가 일치하지 않는다"를 입력으로 받으면 정확하게 예측함. 즉 모델이 훈련셋에 오버피팅이 된 상태임을 확인할 수 있다
- *실제로는 별거없네* (훈련 셋에 포함되지 않은 정의)
```
### desc: 실제로는 별거없네 ###
0: ('산넘어 산', 0.9976289868354797)
1: ('갈수록 태산', 0.002168289152905345)
2: ('꿩 대신 닭', 0.00020149812917225063)
3: ('핑계 없는 무덤 없다', 9.218800869348343e-07)
4: ('등잔 밑이 어둡다', 1.6546708536679944e-07)
5: ('가는 날이 장날', 1.0126942839860931e-07)
6: ('원숭이도 나무에서 떨어진다', 9.898108288552976e-08)
7: ('소문난 잔치에 먹을 것 없다', 6.846833322526891e-09)
8: ('고래 싸움에 새우 등 터진다', 4.417973487047533e-10)
9: ('서당개 삼 년이면 풍월을 읊는다', 8.048845877989264e-14)
```
- *소문과 실제가 일치하지 않는다* (훈련 셋에 포함된 정의)
```
### desc: 소문과 실제가 일치하지 않는다. ###
0: ('소문난 잔치에 먹을 것 없다', 0.999997615814209)
1: ('등잔 밑이 어둡다', 1.7779053678168566e-06)
2: ('가는 날이 장날', 5.957719508842274e-07)
3: ('갈수록 태산', 9.973800452200976e-09)
4: ('핑계 없는 무덤 없다', 2.4250623731347787e-09)
5: ('고래 싸움에 새우 등 터진다', 5.40873457133273e-10)
6: ('산넘어 산', 4.573414147390764e-10)
7: ('원숭이도 나무에서 떨어진다', 2.8081562075676914e-10)
8: ('꿩 대신 닭', 2.690336287081152e-10)
9: ('서당개 삼 년이면 풍월을 읊는다', 3.8126671958460534e-11)
```
- *소문이랑 다르네* ("소문"이라는 단어에는 민감하게 반응한다.) 
```
### desc: 소문이랑 다르네 ###
0: ('산넘어 산', 0.9770968556404114)
1: ('소문난 잔치에 먹을 것 없다', 0.01917330175638199)
2: ('갈수록 태산', 0.0035712094977498055)
3: ('꿩 대신 닭', 8.989872731035575e-05)
4: ('가는 날이 장날', 6.370477785822004e-05)
5: ('핑계 없는 무덤 없다', 1.7765859183782595e-06)
6: ('원숭이도 나무에서 떨어진다', 1.6799665445432765e-06)
7: ('등잔 밑이 어둡다', 1.6705245116099832e-06)
8: ('고래 싸움에 새우 등 터진다', 3.0059517541758396e-08)
9: ('서당개 삼 년이면 풍월을 읊는다', 4.33282611178587e-11)
```

## References
- Devlin,  J. Cheng, M. Lee, K. Toutanova, K. (2018). *: Pre-training of Deep Bidirectional Transformers for Language Understanding*. 
- Gururangan, S. Marasović, A. Swayamdipta, S. Lo, K. Beltagy, I. Downey, D. Smith, N. (2020). *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*
- Hinton, G. Vinyals, O. Dean, J. (2015). *Distilling the Knowledge in a Neural Network*
- Junbum, L. (2020). *KcBERT: Korean Comments BERT*
- Yan, H. Li, X. Qiu, X. Deng, B. (2020). *BERT for Monolingual and Cross-Lingual Reverse Dictionary*


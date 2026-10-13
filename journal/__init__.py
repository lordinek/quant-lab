"""
Elle yapilan islemlerin gunlugu - ve botun o andaki goruşuyle karsilastirmasi.

Neden var: elle trade edildiginde "ne kadar kaybettim"
sorusunun cevabi broker ekraninda var; eksik olan **neden** kaybettigi. Bu
gunluk o soruya iki taraftan bakiyor:

  1. Kendi kaydin - plan neydi, ne oldu, o an ne hissettin.
  2. Botun ayni andaki gorusu - sinyal var miydi, ayni yonde miydi.

Ikincisi bu gunlugu siradan bir Excel tablosundan ayiran sey. Bot zaten her
5 dakikada bir ne dusundugunu `live/paper_history.jsonl`'a yaziyor; elle acilan
bir islem o kayitla eslestirilince su soru olculebilir hale geliyor:

    "Botun onayladigi islemlerde mi kaybediyorum, onaylamadiklarinda mi?"

Cevap ikincisiyse sorun stratejide degil sabirda; birincisiyse strateji
gozden gecirilmeli. Ikisi bambaska mudahale gerektirir ve bugun hangisinde
oldugumuz BILINMIYOR.
"""

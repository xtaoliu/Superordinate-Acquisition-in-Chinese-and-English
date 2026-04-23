"""
Further expansion of the domain set beyond Choe & Papafragou (2026)'s seven.
Motivation: in the Choe seven, 'animal' basics are all single Chinese
characters (猫, 狗, 鸟, 鱼, ...) which Xu (2021) excludes by design. We
therefore add additional superordinate classes whose canonical basic-level
members are typically disyllabic, so that the basic-vs-superordinate AoA
comparison can actually be tested in Xu.
"""
import pandas as pd, numpy as np
from scipy import stats

xu = pd.read_excel("/mnt/project/2021Xu.xlsx")
zw = pd.read_excel("/mnt/project/2023ZhangWord.xlsx", sheet_name="Data")
zw.columns = [c.strip() for c in zw.columns]
zc = pd.read_excel("/mnt/project/2023ZhangCharacter.xlsx", sheet_name="Data")
zc.columns = [c.strip() for c in zc.columns]

# -----------------------------------------------------------------
# Extended superordinate inventory (original 7 + 8 additional classes)
# For each, basic-level members are drawn from modern dictionaries &
# culturally central Chinese referents for that category.
# -----------------------------------------------------------------
CLASSES = {
    # --- original Choe (2026) seven ---
    "toy":          {"sup":"玩具",
        "basics":["娃娃","玩偶","风筝","积木","陀螺","气球","洋娃娃","玩具车"]},
    "animal":       {"sup":"动物",
        "basics":["猫","狗","鸟","鱼","马","牛","羊","虎","熊","兔","鹿","鸭","鸡","象","蛇","龙"]},
    "tool":         {"sup":"工具",
        "basics":["锤子","刀","剪刀","钳子","斧头","扳手","锯子","螺丝刀"]},
    "building":     {"sup":"建筑",
        "basics":["医院","学校","商店","教堂","寺庙","城堡","宫殿","博物馆","图书馆","车站","机场"]},
    "fruit":        {"sup":"水果",
        "basics":["苹果","香蕉","橘子","草莓","葡萄","梨","桃","西瓜","橙子","菠萝","柠檬","樱桃"]},
    "vegetable":    {"sup":"蔬菜",
        "basics":["白菜","萝卜","黄瓜","茄子","番茄","土豆","辣椒","胡椒","青椒","洋葱","菠菜"]},
    "dessert":      {"sup":"甜点",
        "basics":["蛋糕","饼干","布丁","冰淇淋","巧克力","糖果","薄饼","蛋挞","月饼"]},
    # --- additional Chinese-native superordinate classes ---
    "vehicle":      {"sup":"交通工具",
        "basics":["汽车","火车","飞机","轮船","自行车","公交车","地铁","摩托车","卡车"]},
    "furniture":    {"sup":"家具",
        "basics":["桌子","椅子","沙发","衣柜","床铺","书桌","橱柜","凳子","茶几"]},
    "appliance":    {"sup":"电器",
        "basics":["冰箱","电视","洗衣机","空调","微波炉","电饭锅","电风扇","吹风机"]},
    "stationery":   {"sup":"文具",
        "basics":["铅笔","钢笔","橡皮","尺子","书包","文具盒","墨水","水彩"]},
    "clothing":     {"sup":"衣服",
        "basics":["衬衫","裤子","裙子","外套","毛衣","T恤","围巾","帽子"]},
    "instrument":   {"sup":"乐器",
        "basics":["钢琴","小提琴","吉他","鼓","笛子","二胡","琵琶","口琴"]},
    "beverage":     {"sup":"饮料",
        "basics":["牛奶","咖啡","果汁","啤酒","茶水","汽水","可乐","豆浆"]},
    "flower":       {"sup":"花卉",
        "basics":["玫瑰","菊花","牡丹","荷花","向日葵","郁金香","百合","茉莉"]},
}

# Also test 'food' 食物 which Choe (2026) footnote 1 mentioned
CLASSES["food"] = {"sup":"食物",
    "basics":["米饭","面条","馒头","饺子","面包","包子","炒饭","粥"]}

# Collect everything
rows = []
for dom, meta in CLASSES.items():
    sup = meta["sup"]
    m = xu[xu["Word"]==sup]
    if len(m):
        rows.append({"domain":dom,"level":"Superordinate",
                     "chinese":sup,"AoA":float(m["AoA Mean"].iloc[0]),
                     "n_char":len(sup)})
    for b in meta["basics"]:
        m = xu[xu["Word"]==b]
        if len(m):
            rows.append({"domain":dom,"level":"Basic",
                         "chinese":b,"AoA":float(m["AoA Mean"].iloc[0]),
                         "n_char":len(b)})

ext = pd.DataFrame(rows)
ext.to_csv("/home/claude/analysis/further_expanded.csv", index=False, encoding="utf-8-sig")

def h(t): print(f"\n{'='*70}\n {t}\n{'='*70}")

h("C1. Coverage of each domain in Xu (2021), including new superordinate classes")
cov = (ext.groupby(["domain","level"])["chinese"].count()
         .unstack(fill_value=0)
         .rename(columns={"Basic":"n_basics_in_Xu","Superordinate":"sup_in_Xu"}))
# Note which domains have NO basics in Xu (because all basics are monosyllabic)
monosyll_only = []
for dom, meta in CLASSES.items():
    if cov.loc[dom,"n_basics_in_Xu"] == 0 if dom in cov.index else True:
        # count monosyllabic basics
        mono = [b for b in meta["basics"] if len(b)==1]
        if len(mono):
            monosyll_only.append((dom, mono))
cov = cov.reindex([d for d in CLASSES if d in cov.index])
print(cov.to_string())
print("\nDomains where ALL listed basics are monosyllabic (and therefore excluded from Xu by design):")
for dom, mono in monosyll_only:
    print(f"  {dom}: {' '.join(mono)}")

h("C2. Subjective AoA (Xu 2021): Superordinate vs. Basic, across the expanded set")
testable = ext.copy()
by_dom = testable.groupby(["domain","level"])["AoA"].mean().unstack()
by_dom = by_dom.dropna()
print(f"\nDomains with both levels represented in Xu: {len(by_dom)}\n")
print(by_dom.round(2).sort_values("Superordinate"))

tp,pp = stats.ttest_rel(by_dom["Superordinate"], by_dom["Basic"])
w = stats.wilcoxon(by_dom["Superordinate"], by_dom["Basic"])
diff = (by_dom["Superordinate"]-by_dom["Basic"])
print(f"\nMean Superordinate AoA: {by_dom['Superordinate'].mean():.2f}")
print(f"Mean Basic AoA       : {by_dom['Basic'].mean():.2f}")
print(f"Mean diff (Sup - Basic): {diff.mean():.3f}  (negative = superordinate earlier)")
print(f"Paired t({len(by_dom)-1}) = {tp:.3f}, p = {pp:.4f}")
print(f"Wilcoxon W = {w.statistic:.1f}, p = {w.pvalue:.4f}")
print(f"Sign: Sup earlier in {(diff<0).sum()} / {len(diff)} domains")

h("C3. Item-level comparison on the expanded set")
s = ext.loc[ext["level"]=="Superordinate","AoA"]
b = ext.loc[ext["level"]=="Basic",        "AoA"]
t,p = stats.ttest_ind(s, b, equal_var=False)
u,pu = stats.mannwhitneyu(s,b,alternative="two-sided")
print(f"\nn_sup = {len(s)}, n_basic = {len(b)}")
print(f"Welch t = {t:.3f}, p = {p:.4f}")
print(f"Mann-Whitney U = {u:.1f}, p = {pu:.4f}")
print(f"Sup mean AoA: {s.mean():.2f} (SD {s.std():.2f})")
print(f"Basic mean AoA: {b.mean():.2f} (SD {b.std():.2f})")

h("C4. Linear mixed-effects substitute: by-domain means")
# Since small n per domain, use the domain means.
dm = by_dom.reset_index()
from scipy.stats import pearsonr
print("\nBy-domain scatter:")
print(dm.round(2))


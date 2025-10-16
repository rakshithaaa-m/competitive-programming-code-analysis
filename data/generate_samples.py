# data/generate_samples.py
import os

os.makedirs("data/snippets", exist_ok=True)

# Simple C++ code snippets representing different styles:
samples = [
    ("newbie_01.cpp", """
#include <iostream>
using namespace std;
int main() {
    int n; 
    cin >> n;
    cout << n << endl;
    return 0;
}
"""),
    ("expert_01.cpp", """
#include <bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n; scanf("%d", &n);
    vector<int> a(n);
    for (int i=0;i<n;i++) scanf("%d",&a[i]);
    sort(a.begin(), a.end());
    for (int x: a) printf("%d\\n", x);
    return 0;
}
"""),
    ("expert_02.cpp", """
#include <bits/stdc++.h>
using namespace std;
template<typename T> T maxv(T a, T b){ return a>b?a:b; }
int main(){
    int n; scanf("%d", &n);
    long long sum=0;
    for(int i=0;i<n;i++){ int x; scanf("%d",&x); sum+=x; }
    cout<<sum<<\"\\n\";
    return 0;
}
"""),
    ("mid_01.cpp", """
#include <iostream>
#include <vector>
using namespace std;
int main(){
    int n; cin>>n;
    vector<int> v(n);
    for(int i=0;i<n;i++) cin>>v[i];
    sort(v.begin(), v.end());
    for(int i=0;i<n;i++) cout<<v[i]<<\" \";
    return 0;
}
"""),
    # add more variations...
]

# write snippets
for fname, content in samples:
    with open(os.path.join("data/snippets", fname), "w", encoding="utf-8") as f:
        f.write(content)

# create labels.csv: label convention (for demo):
# 0: Newbie, 1: Pupil, 2: Specialist, 3: Expert, 4: Candidate Master, 5: Master, 6: International Master, 7: Grandmaster, 8: International Grandmaster, 9: Legendary Grandmaster
# We'll just map examples to three coarse classes for clarity: 0=newbie,1=mid,2=expert
labels = [
    ("newbie_01.cpp", 0),
    ("mid_01.cpp", 1),
    ("expert_01.cpp", 2),
    ("expert_02.cpp", 2),
]

with open("data/labels.csv", "w", encoding="utf-8") as f:
    f.write("filename,label\n")
    for fn, lab in labels:
        f.write(f"{fn},{lab}\n")

print("Created sample snippets in data/snippets and labels.csv")


import pickle

files = 'metrics_fixed_'
dates=['20191012','20191117','20191223','20200116','20200221','20200316',
					'20200421','20200515','20200620','20200714','20200819','20200912']

for date in dates:
    print("date",date)
    f = open(files+date+'.pkl', "rb")
    output = pickle.load(f)
    print(output)

print("Small classes...")
for date in dates:
    print("date",date)
    f = open(files+date+'_small_classes.pkl', "rb")
    output = pickle.load(f)
    print(output)
    
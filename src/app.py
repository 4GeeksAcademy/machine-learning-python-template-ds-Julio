from utils import db_connect
engine = db_connect()

# your code here
import flask from Flask , render_templete, request
import pickle

app =Flask(_name_)

cv=pickle.load(open('/workspaces/machine-learning-python-template-ds-Julio/models/cv.pkl','rb'))
clf=pickle.load(open('/workspaces/machine-learning-python-template-ds-Julio/models/clf.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['post'])
def predict():
    email=request.form.get('email')
    print(email)
    x=cv.transform([email])
    prediction=clf.predict(x)
    prediction=1 if prediction==1 else -1 
    return render_template('index.html',response=prediction)



if __name__ =="_main_":
    app.run(debug=True)



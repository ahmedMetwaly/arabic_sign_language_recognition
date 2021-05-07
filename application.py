


app = Flask(__name__)
with open('signs4.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
@app.home('/home')
def home():
  return "hello"

if __name__ == "__main__":
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run()

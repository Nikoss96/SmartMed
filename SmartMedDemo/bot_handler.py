import telebot
import config
import random

from telebot import types

bot_token = '6586604489:AAGbck2PsvKCm3SeQ7Y5Xf6-agRtgbxf1xk'
bot = telebot.TeleBot(bot_token)

@bot.message_handler(commands=['start'])
def welcome(message):
    #img = open('static/welcome.gif', 'rb')
    bot.send_message(message.chat.id, text = "hui")

    # keyboard
    markup = types.InlineKeyboardMarkup(row_width=2)
    item1 =types.InlineKeyboardButton("", callback_data='TEST1')
    item2 =types.InlineKeyboardButton("", callback_data='TEST10')
    item3 =types.InlineKeyboardButton("", callback_data='TEST10')
    item4 =types.InlineKeyboardButton("",'')
    item5 =types.InlineKeyboardButton("👥קהילה ראשית👥", '')
    item6 =types.InlineKeyboardButton("", callback_data='TEST4')
    item7 =types.InlineKeyboardButton("", callback_data='TEST5')
    item8 =types.InlineKeyboardButton("📚ביקורות📚", callback_data='TEST6')
    item9 =types.InlineKeyboardButton("♻️שיתוף♻️", callback_data='TEST7')
    item10 =types.InlineKeyboardButton("👤בדיקת סוחר👤", callback_data='TEST8')
    item11 =types.InlineKeyboardButton("❓מי אנחנו❓", callback_data='TEST9')

    markup.add(item1, item2 ,item3 ,item4 ,item5 ,item6 ,item7 ,item8 ,item9 ,item10 ,item11)


    bot.send_message(message.chat.id, "שלום🤚, {0.first_name}!\nטלקוקאין נוסדה בשנת 2020 , על ידי צוות מתכנתים מהשורה הראשונה 🛠 🔐בדגש על אבטחת הנתונים ❗️.\nאנחנו כאן בשביל להנגיש סמים בכל הארץ.\nאוהבים 💎TeleCocaine💎 רכישה מהנה.".format(message.from_user, bot.get_me()),
        parse_mode='html', reply_markup=markup)





@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    try:
        if call.message:
            if call.data == 'TEST9':
                bot.send_message(call.message.chat.id, 'אז בקצרה...Telecocaine ישראל מתעסק בתיווך🔀\nבין סוחרים ללקוחות אנחנו דואגים ללקוחות\nסוחרים ברמה הגבוה ביותר שרק אפשר לדמיין\nשירות לקוחות ומענה אנושי 24/7☎️\nעזרה לכול נושא או בעיה יש למענכם צוות\n👨🏻‍💻מנהלים אדיב אמין ומסור\nושירותי בוטים מאובטחים 🔐')







    except Exception as e:
        print(repr(e))

# RUN
bot.polling(none_stop=True)
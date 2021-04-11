# -*- coding: utf-8 -*-

from selectolax.parser import HTMLParser

def get_text_from_html(html):
    '''
    Extracted from https://rushter.com/blog/python-fast-html-parser/ to eliminate html tags from email body

    Parameters
    html: html file
    Return
    text: html text content
    '''
    tree = HTMLParser(html)

    if tree.body is None:
        return html

    for tag in tree.css('script'):
        tag.decompose()
    for tag in tree.css('style'):
        tag.decompose()

    text = unidecode.unidecode(tree.body.text(separator=' '))

    return text

def clean_subject(mail_header):
    '''
    Clean subject
    Parameters
    mail_header: email.Header object or string or None.
    Return
    decoded_header: string containing mail subject
    '''

    if type(mail_header) == Header:
        decoded_header = decode_header(mail_header.encode())[0][0].decode('utf-8')
    else:
        decoded_header = mail_header

    if decoded_header[:5] == 'Fwd: ':
        decoded_header = decoded_header[5:]
    elif decoded_header[:4] == 'Re: ':
        decoded_header = decoded_header[4:]

    decoded_header = re.sub(r"[^a-zA-Z?.!,¿]+", " ", decoded_header)

    return decoded_header

def clean_body(mail_body):
    '''
    Clean mail body
    Parameters
    mail_body: string containing the body of a mail
    Return
    Filtered mail body
    '''
    mail_body = mail_body.replace('\t', '')
    filtered_spanish = regex.split(r'(\bEl \d{1,2} de [a-z]+ de 2\d{3},)', mail_body)[0]
    filtered_english = regex.split(r'(\bOn \d{1,2} de [a-z]+. de 2\d{3},)', filtered_spanish)[0]

    if len(filtered_english) > 0:
        return filtered_english
    else:
        return filtered_spanish

# Load complete email data
mails = pk.load(open("../../../text_data/mails.txt", 'rb'))
columns_names = ['id','subject','date','sender','recipient','body']
df = pd.DataFrame(data=mails, columns=columns_names)

# Filter text from hltm emails
df['body'] = df['body'].apply(get_text_from_html)

# Extract sender and recipient email only
email_regex = "([a-z0-9-.]+@[a-z0-9-.]+)"
df['sender_email'] = df.sender.str.lower().str.extract(email_regex)[0]
df['recipient_email'] = df.recipient.str.lower().str.extract(email_regex)[0]

# Eliminate 'no reply' and 'automatic' msgs
df_noreply = df[~df.sender.str.contains('noreply@google.com').fillna(False)]
df_noautom = df_noreply[~df_noreply.subject.str.contains('Respuesta automática').fillna(False)]

# Separate msgs by type of sender
alumns_filter = df_noautom.sender.str.contains('@alum.up.edu.pe').fillna(False)
send_by_alumns = df_noautom[alumns_filter]
send_by_no_alumns = df_noautom[~alumns_filter]
send_by_internals = df_noautom[df_noautom.sender.str.contains('@up.edu.pe').fillna(False)]

print('# msgs send by alumns:', len(send_by_alumns))
print('# of alumns that send msgs:', len(send_by_alumns.sender_email.unique()))

# Clean mails subject
send_by_internals['subject'] = send_by_internals['subject'].apply(clean_subject)

# Separate mails sended to each alumn
dfs = [send_by_internals[send_by_internals.recipient_email == alumn] for alumn in send_by_alumns.sender_email.unique()]

unique_alumns = send_by_alumns.sender_email.unique()
send_by_internals[send_by_internals["recipient_email"].isin(unique_alumns)]

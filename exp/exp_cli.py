from prompt_toolkit import prompt
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.styles import style_from_dict
from prompt_toolkit.token import Token


def help_cmd():
    print("This ia a help command!")


CMDs = {
    "help": help_cmd
}


# cmd_completer = WordCompleter(['help'])

class CmdSuggest(AutoSuggest):

    def get_suggestion(self, cli, buffer, document):
        text = document.text.lstrip(' ')
        # print(text)
        text = text.rsplit('\n', 1)[-1]
        for cmd in CMDs.keys():
            if cmd.startswith(text):
                return Suggestion(cmd[len(text):])


def get_bottom_toolbar_tokens(cli):
    return [(Token.Toolbar, 'Welcome to GridX > type help for command list')]


style = style_from_dict({
    Token.Toolbar: '#ffffff bg:#CCCCCC',
})


def init():
    while True:
        text = prompt('> ', get_bottom_toolbar_tokens=get_bottom_toolbar_tokens,
                      style=style,
                      auto_suggest=CmdSuggest())
        print('You said: %s' % text)
        if text in CMDs:
            CMDs[text]()

        # prompt('What is your name: ', get_bottom_toolbar_tokens=get_bottom_toolbar_tokens, style=style)
        # prompt('What is your name: ')


if __name__ == "__main__":
    init()

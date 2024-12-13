If you open the advanced options in the character creation area then you'll see the "custom code" input. This allows you to add some JavaScript code that extend the functionality of your character.

Some examples of what you can do with this:

 * Give your character access to the internet (e.g. so you can ask it to summarise webpages)
 * Improve your character's memory by setting up your own embedding/retrieval system (see "Storing Data" section below) 
 * Give your character a custom voice using an API like [ElevenLabs](https://api.elevenlabs.io/docs)
 * Allow your character to run custom JS or [Python](https://github.com/josephrocca/OpenCharacters/blob/main/docs/running-python-code.md) code
 * Give your character the ability to create pictures using Stable Diffusion
 * [Auto-delete/retry messages](https://github.com/josephrocca/OpenCharacters/blob/main/docs/custom-code-examples.md#add-a-refinement-step-to-the-messages-that-your-character-generates) from your character that contain certain keywords
 * Change the background image of the chat, or the chat bubble style, or the avatar images, or the music, depending on what's happening in your story


## Examples

After reading this doc to get a sense of the basics, visit this page for more complex, "real-world" examples: [custom-code-examples.md](https://github.com/josephrocca/OpenCharacters/blob/main/docs/custom-code-examples.md)

## The `oc` Object

Within your custom code, you can access and update `oc.thread.messages`. It's an array that looks like this:
```json5
[
  {
    author: "user",
    name: "Anon",
    hiddenFrom: [], // can contain "user" and/or "ai"
    content: "Hello",
  },
  {
    author: "ai",
    name: "Levi Ackerman",
    hiddenFrom: [],
    content: "Hi.",
  },
  {
    author: "system",
    name: "System",
    hiddenFrom: ["user"],
    expectsReply: false, // this means the AI won't automatically reply to this message
    content: "Here's an example system message that's hidden from the user and which the AI won't automatically reply to.",
  },
]
```
The most recent message is at the bottom/end of the array. The `author` field can be `user`, `ai`, or `system`. Use "system" for guiding the AI's behavior, and including context/info where it wouldn't make sense to have that context/info come from the user or the AI.

Below is an example that replaces `:)` with `૮ ˶ᵔ ᵕ ᵔ˶ ა` in every message that is added to the thread. Just paste it into the custom code box to try it out.
```js
oc.thread.on("MessageAdded", function({message}) {
  message.content = message.content.replaceAll(":)", "૮ ˶ᵔ ᵕ ᵔ˶ ა");
});
```
You can edit existing messages like in this example, and you can also delete them by just removing them from the `oc.thread.messages` array (with `pop`, `shift`, `splice`, or however else), and you can of course add new ones - e.g. with `push`/`unshift`.

Note that your `MessageAdded` handler can be `async`, and it'll be `await`ed so that you can be sure your code has finished running before the AI responds.

You can also access and edit character data via `oc.character.propertyName`. Here's a full list of all the property names that you can access and edit on the `oc` object:

  * `character`
    * `name`
    * `avatar`
      * `url` - url to an image
      * `size` - multiple of default size
      * `shape` - "circle" or "square" or "portrait" 
    * `roleInstruction`
    * `reminderMessage`
    * `initialMessages`
    * `customCode` - yep, a character can edit its own custom code
    * `temperature`
    * `topP`
    * `frequencyPenalty`
    * `presencePenalty`
    * `stopSequences`
    * `modelName`
    * `streamingResponse`
    * `customData` - an object/dict where you can store arbitrary data
      * `PUBLIC` - a special sub-property of `customData` that will be shared within character sharing URLs
 * `thread`
   * `name`
   * `messages` - an **array** of messages, where **each message** has:
     * `content` - the message text - it can include HTML, and is rendered as [markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) by default (see `oc.messageRenderingPipeline`)
     * `author`
     * `name`
     * `hiddenFrom` - array that can contain "user" or "ai" or both or neither
     * `expectsReply` - `true` (bot will reply to this message) or `false` (bot will not reply), or `undefined` (use default behavior - i.e. reply to user messages, but not own messages)
     * `customData` - message-specific custom data storage
     * `avatar` - this will override the user's/ai's default avatar for this particular message
       * `url`
       * `size`
       * `shape`
     * `wrapperStyle` - css for the "message bubble" - e.g. "background:white; border-radius:10px; color:grey;"
       * note that you can include HTML within the `content` of message (but you should use `oc.messageRenderingPipeline` for visuals where possible - see below)
     * `instruction` - the instruction that was written in `/ai <instruction>` or `/user <instruction>` - used when the regenerate button is clicked
     * `scene` - the most recent message that has a scene is the scene that is "active"
       * `background`
         * `url` - image or video url
         * `filter` - [css filter](https://developer.mozilla.org/en-US/docs/Web/CSS/filter) - e.g. `hue-rotate(90deg); blur(5px)`
       * `music`
         * `url` - audio url (also supports video urls)
         * `volume` - between 0 and 1
   * `character` - thread-specific character overrides
     * `name`
     * `avatar`
       * `url`
       * `size`
       * `shape`
     * `reminderMessage`
     * `roleInstruction`
   * `userCharacter` - thread-specific user character overrides
     * `name`
     * `avatar`
       * `url`
       * `size`
       * `shape`
   * `systemCharacter` - thread-specific system character overrides
     * `name`
     * `avatar`
       * `url`
       * `size`
       * `shape`
   * `customData` - thread-specific custom data storage
   * `messageWrapperStyle` - CSS applied to all messages in the thread, except those with `message.wrapperStyle` defined
 * `messageRenderingPipeline` - an array of processing functions that get applied to messages before they are seen by the user and/or the ai (see "Message Rendering" section below)

Note that many character properties aren't available in the character editor UI, so if you e.g. wanted to add a stop sequence for your character so it stops whenever it writes ":)" and also set presence pentalty to 1, then you could do it by adding this text to the custom code text box in the character editor:
```js
oc.character.stopSequences = [":)"];
oc.character.presencePenalty = 1;
```

Here's some custom code which allows the AI to see the contents of webpages/PDFs if you put URLs in your messages:

```js
async function getPdfText(data) {
  let doc = await window.pdfjsLib.getDocument({data}).promise;
  let pageTexts = Array.from({length: doc.numPages}, async (v,i) => {
    return (await (await doc.getPage(i+1)).getTextContent()).items.map(token => token.str).join('');
  });
  return (await Promise.all(pageTexts)).join(' ');
}
      
oc.thread.on("MessageAdded", async function ({message}) {
  if(message.author === "user") {
    let urlsInLastMessage = [...message.content.matchAll(/https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)/g)].map(m => m[0]);
    if(urlsInLastMessage.length === 0) return;
    if(!window.Readability) window.Readability = await import("https://esm.sh/@mozilla/readability@0.4.4?no-check").then(m => m.Readability);
    let url = urlsInLastMessage.at(-1); // we use the last URL in the message, if there are multiple
    let blob = await fetch(url).then(r => r.blob());
    let output;
    if(blob.type === "application/pdf") {
      if(!window.pdfjsLib) {
        window.pdfjsLib = await import("https://cdn.jsdelivr.net/npm/pdfjs-dist@3.6.172/+esm").then(m => m.default);
        pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.6.172/build/pdf.worker.min.js";
      }
      let text = await getPdfText(await blob.arrayBuffer());
      output = text.slice(0, 5000); // <-- grab only the first 5000 characters (you can change this)
    } else {
      let html = await blob.text();
      let doc = new DOMParser().parseFromString(html, "text/html");
      let article = new Readability(doc).parse();
      output = `# ${article.title || "(no page title)"}\n\n${article.textContent}`;
      output = output.slice(0, 5000); // <-- grab only the first 5000 characters (you can change this)
    }
    oc.thread.messages.push({
      author: "system",
      hiddenFrom: ["user"], // hide the message from user so it doesn't get in the way of the conversation
      content: "Here's the content of the webpage that was linked in the previous message: \n\n"+output,
    });
  }
});
```

Custom code is executed securely (i.e. in a sandboxed iframe), so if you're using a character that was created by someone else (and that has some custom code), then their code won't be able to access your OpenAI API key, or your messages with other characters, for example. The custom code only has access to the character data and the messages for your current conversation.

Here's some custom code that adds a `/charname` command that changes the name of the character. It intercepts the user messages, and if it begins with `/charname`, then it changes `oc.character.name` to whatever comes after `/charname`, and then deletes the message.
```js
oc.thread.on("MessageAdded", async function ({message}) {
  let m = message; // the message that was just added
  if(m.author === "user" && m.content.startsWith("/charname ")) {
    oc.character.name = m.content.replace(/^\/charname /, "");
    oc.thread.messages.pop(); // remove the message
  }
});
```

### Events

Each of these events has a `message` object, and `MessageDeleted` has `originalIndex` for the index of the deleted message:

 * `oc.thread.on("MessageAdded", function({message}) { ... })` - a message was added to the end of the thread
 * `oc.thread.on("MessageEdited", function({message}) { ... })` - message was edited or regenerated
 * `oc.thread.on("MessageInserted", function({message}) { ... })` - message was inserted (see message editing popup)
 * `oc.thread.on("MessageDeleted", function({message, originalIndex}) { ... })` - user deleted a message (trash button)

The `message` object is an actual reference to the object, so you can edit it directly like this:

```js
oc.thread.on("MessageAdded", function({message}) {
  message.content += "blah";
})
```

Here's an example of how you can get the *index* of edited messages:

```js
oc.thread.on("MessageEdited", function({message}) {
  let editedMessageIndex = oc.thread.messages.findIndex(m => m === message);
  // ...
});
```



### Message Rendering
Sometimes you may want to display different text to the user than what the AI sees. For that, you can use `oc.messageRenderingPipeline`. It's an array that you `.push()` a function into, and that function is used to process messages. Your function should use the `reader` parameter to determine who is "reading" the message (either `user` or `ai`), and then "render" the message `content` accordingly. Here's an example to get you started:
```js
oc.messageRenderingPipeline.push(function({message, reader}) {
  if(reader === "user") message.content += "🌸"; // user will see all messages with a flower emoji appended
  if(reader === "ai") message.content = message.content.replaceAll("wow", "WOW"); // ai will see a version of the message with all instances of "wow" capitalized
});
```

### Visual Display and User Inputs

Your custom code runs inside an iframe. You can visually display the iframe using `oc.window.show()` (and hide with `oc.window.hide()`). The user can drag the embed around on the page and resize it. All your custom code is running within the iframe embed whether it's currently displayed or not. You can display content in the embed by just executing custom code like `document.body.innerHTML = "hello world"`.

You can use the embed to e.g. display a dynamic video/gif avatar for your character that changes depending on the emotion that is evident in the characters messages ([example](https://github.com/josephrocca/OpenCharacters/blob/main/docs/custom-code-examples.md#append-image-based-on-predicted-facial-expression-of-the-message)). Or to e.g. display the result of the p5.js code that the character is helping you write. And so on.

### Using the GPT API in Your Custom Code

You may want to use GPT/LLM APIs in your message processing code. For example, you may want to classify the sentiment of a message in order to display the correct avatar (see "Visual Display ..." section), or you may want to implement your own custom chat-summarization system, for example. In this case, you can use `oc.getChatCompletion`.

Use it like this:
```js
let result = await oc.getChatCompletion({
  messages: [{author:"system", content:"..."}, {author:"user", content:"..."}, {author:"ai", content:"..."}, ...],
  temperature: 1,
  stopSequences: ["\n"],
  ...
});
```
The `messages` parameter is the only required one.

Here's an example of some custom code that edits all messages to include more emojis:

```js
oc.thread.on("MessageAdded", async function({message}) {
  let result = await oc.getChatCompletion({
    messages: [{author:"user", content:`Please edit the following message to have more emojis:\n\n---\n${message.content}\n---\n\nReply with only the above message (the content between ---), but with more (relevant) emojis.`}],
  });
  message.content = result.trim().replace(/^---|---$/g, "").trim();
});
```

## Storing Custom Data

If you'd like to save some data that is generated by your custom code, then you can do that by using `oc.thread.customData` - e.g. `oc.thread.customData.foo = 10`. You can also store custom data on individual messages like this: `message.customData.foo = 10`. If you want to store data in the character itself, then use `oc.character.customData.foo = 10`, but note that this data will not be shared within character share links. If you *do* want to save the data to the character in a way that's preserved in character share links, then you should store data under `oc.character.customData.PUBLIC` - e.g. `oc.character.customData.PUBLIC = {foo:10}`.


## Streaming Messages
See the [text-to-speech plugin code](https://github.com/josephrocca/OpenCharacters/blob/main/plugins/tts/main.js) for a "real-world" example of this.
```js
oc.thread.on("StreamingMessage", async function (data) {
  for await (let chunk of data.chunks) {
    console.log(chunk.text); // `chunk.text` is a small fragment of text
  }
});
```

## Interactive Messages
You can use button `onclick` handlers in message so that e.g. the user can click a button to take an action instead of typing:
```html
What would you like to do?
1. <button onclick="oc.thread.messages.push({author:'user', content:'Fight'});">Fight</button>
2. <button onclick="oc.thread.messages.push({author:'user', content:'Run'});">Run</button>
```
I recommend that you use `oc.messageRenderingPipeline` to turn a custom format into HTML, rather than actually having HTML in your messages (the HTML would use more tokens, and might confuse the AI). So your format might look like this:
```html
What would you like to do?
1. [[Fight]]
2. [[Run]]
```
You could prompt/instruct/remind your character to reply in that format with an instruction message that's something similar to this:
```
You are a game master. You creatively and engagingly simulate a world for the user. The user takes actions, and you describe the consequences.

Your messages should end with a list of possible actions, and each action should be wrapped in double-square brackets like this:

Actions:
1. [[Say sorry]]
2. [[Turn and run]]
```
And then you'd add this to your custom code:
```js
oc.messageRenderingPipeline.push(function({message, reader}) {
  if(reader === "user") {
    message.content = message.content.replace(/\[\[(.+?)\]\]/g, (match, text) => {
      let encodedText = encodeURIComponent(text); // this is a 'hacky' but simple way to prevent special characters like quotes from breaking the onclick attribute
      return `<button onclick="oc.thread.messages.push({author:'user', content:decodeURIComponent('${encodedText}')});">${text}</button>`;
    });
  }
});
```
If you want to change something about the way this works (e.g. change the double-square-bracket format to something else), but don't know JavaScript, the "Custom Code Helper" starter character might be able to help you make some adjustments.

Note that you can't use the `this` keyword within the button onclick handler - it actually just sends the code in the onclick to your custom code iframe and executes it there, so there's no actual element that's firing the onclick from the iframe's perspective, and thus no `this` or `event`, etc.

## Gotchas

### "&lt;function&gt; is not defined" in click/event handlers
The following code won't work:
```js
function hello() {
  console.log("hi");
}
document.body.innerHTML = `<div onclick="hello()">click me</div>`;
oc.window.show();
```
This is because all custom code is executed inside a `<script type=module>` so you need to make functions *global* if you want to access them from *outside* the module (e.g. in click handlers). So if you want to the above code to work, you should define the `hello` function like this instead:
```js
window.hello = function() {
  console.log("hi");
}
```



# FAQ

* Is it possible to run a custom function before the AI tries to respond? I.e., after the user message lands, but before the AI responds? And then kick off the AI response process after the async call returns?
  * **Answer:** Yep, the `MessageAdded` event runs every time a message is added - user or ai. So you can check `if(oc.thread.messages.at(-1).author === "user") { ... }` (i.e. if latest message is from user) and the `...` code will run right after the user responds, and *before* the ai responds.

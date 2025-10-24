from neomodel import (
    StructuredNode,
    StructuredRel,
    StringProperty,
    IntegerProperty,
    UniqueProperty,
    ArrayProperty,
    DateTimeProperty,
    JSONProperty,
    RelationshipTo
)
# class Cites(StructuredRel):
#     cite_type = StringProperty()

class Container(StructuredRel):
    contain_type = StringProperty()

class PublishedBy(StructuredRel):
    published_time = DateTimeProperty()


class Path(StructuredNode):
    path = StringProperty(UniqueProperty=True)
    embedding = ArrayProperty()

    contain_document = RelationshipTo('Document', 'contain', model=Container)

class Document(StructuredNode):
    doc_name = StringProperty(UniqueProperty=True)
    pub_time = DateTimeProperty()
    doc_type = StringProperty()
    doc_num = StringProperty()
    doc_url = StringProperty()
    doc_id = IntegerProperty(index=True)
    embedding = ArrayProperty()

    contain_title = RelationshipTo('Document', 'contain', model=Container)
    contain_clause = RelationshipTo('Clause', 'contain', model=Container)
    publishedBy = RelationshipTo('PublishedBy', 'publishedBy', model=PublishedBy)
    KWD = RelationshipTo('Keyword', 'KWD')

class Title(StructuredNode):
    doc_title_name = StringProperty()

    contain_title = RelationshipTo('Title', 'contain', model=Container)
    contain_clause = RelationshipTo('Clause', 'contain', model=Container)

class Clause(StructuredNode):
    doc_id = StringProperty(UniqueProperty=True)
    title_name = StringProperty()
    summary = StringProperty()
    embedding = ArrayProperty()

    contain_content = RelationshipTo('Content', 'contain_content', model=Container)
    cites = RelationshipTo('Document', 'cites')

class Content(StructuredNode):
    text_content = StringProperty()
    picture_content = StringProperty() #base64或者之后改为json
    video_content = JSONProperty #用于储存视频介绍和下载链接

class Keyword(StructuredNode):
    keyword = StringProperty()
    embedding = ArrayProperty()

class Organization(StructuredNode):
    institute_name = StringProperty(UniqueProperty=True)
    embedding = ArrayProperty()
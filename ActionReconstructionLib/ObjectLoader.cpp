#include "ObjectLoader.h"
#include <sstream>
#include <fstream>

#define TINYPLY_IMPLEMENTATION
#include "../third-party/tinyply/source/tinyply.h"

using namespace std;

ci::TriMeshRef ObjectLoader::loadCustomObj(const std::string & path)
{
	ci::DataSourceRef dataSource = ci::loadFile(path);
	std::shared_ptr<ci::IStreamCinder> stream(dataSource->createStream());

	std::vector<ci::vec3> positions;
	std::vector<ci::vec3> normals;
	std::vector<ci::ColorAf> colors;
	std::vector<ci::vec2> texCoords0;
	std::vector<ci::vec3> texCoords1;
	std::vector<uint32_t> indices;

	//read data
	size_t lineNumber = 0;
	while (!stream->isEof()) {
		lineNumber++;
		string line = stream->readLine(), tag;
		if (line.empty() || line[0] == '#')
			continue;

		while (line.back() == '\\' && !stream->isEof()) {
			auto next = stream->readLine();
			line = line.substr(0, line.size() - 1) + next;
		}

		stringstream ss(line);
		ss >> tag;
		if (tag == "v") { // vertex
			ci::vec3 v;
			ss >> v.x >> v.y >> v.z;
			//mesh->appendPosition(v);
			positions.push_back(v);
			//maybe also color?
			ss >> v.x;
			if (!ss.eof()) {
				ss >> v.y >> v.z;
				//mesh->appendColorRgba(ci::ColorAf(v.x, v.y, v.z, 1.0f));
				colors.emplace_back(v.x, v.y, v.z, 1.0f);
			}
		}
		else if (tag == "vn") { // vertex normals
			ci::vec3 v;
			ss >> v.x >> v.y >> v.z;
			//mesh->appendNormal(normalize(v));
			normals.push_back(normalize(v));
		}
		else if (tag == "vc") { // vertex colors
			ci::ColorAf v;
			ss >> v.r >> v.g >> v.b >> v.a;
			//mesh->appendColorRgba(v);
			colors.push_back(v);
		}
		else if (tag == "vg") { // cell index
			ci::vec3 v;
			ss >> v.x >> v.y >> v.z;
			//mesh->appendTexCoord1(v);
			texCoords1.push_back(v);
		}
		else if (tag == "vt") { // tex coords
			ci::vec2 v;
			ss >> v.x >> v.y;
			//mesh->appendTexCoord1(v);
			texCoords0.push_back(v);
		}
		else if (tag == "f") { // face
			uint32_t a, b, c;
			ss >> a;
			while (ss.get() != ' ') {};
			ss >> b;
			while (ss.get() != ' ') {};
			ss >> c;
			//mesh->appendTriangle(a - 1, b - 1, c - 1);
			indices.push_back(a - 1);
			indices.push_back(b - 1);
			indices.push_back(c - 1);
		}
	}

	//create the mesh
	ci::TriMesh::Format format = ci::TriMesh::Format().positions(3);
	if (!colors.empty()) format = format.colors(4);
	if (!normals.empty()) format = format.normals();
	if (!texCoords0.empty()) format = format.texCoords0(2);
	if (!texCoords1.empty()) format = format.texCoords1(3);
	ci::TriMeshRef mesh = ci::TriMesh::create(format);

	mesh->appendPositions(positions.data(), positions.size());
	if (!colors.empty()) mesh->appendColors(colors.data(), colors.size());
	if (!normals.empty()) mesh->appendNormals(normals.data(), normals.size());
	if (!texCoords0.empty()) mesh->appendTexCoords0(texCoords0.data(), texCoords0.size());
	if (!texCoords1.empty()) mesh->appendTexCoords1(texCoords1.data(), texCoords1.size());
	mesh->appendIndices(indices.data(), indices.size());

	//if (mesh->getBufferTexCoords1().empty())
	//	mesh->getBufferTexCoords1().resize(3 * mesh->getNumVertices(), 0.0f);

	return mesh;
}

void ObjectLoader::saveCustomObj(ci::TriMeshRef mesh, const std::string & path)
{
	std::ofstream stream(path, std::ofstream::trunc);
	int n = mesh->getNumVertices();
	int m = mesh->getNumTriangles();

	stream << "g SoftBody\n\n";
	//vertices
	if (mesh->hasColorsRgb()) {
		for (int i = 0; i < n; ++i)
			stream << "v " << mesh->getPositions<3>()[i].x << " " << mesh->getPositions<3>()[i].y << " " << mesh->getPositions<3>()[i].z << " " <<
				mesh->getColors<3>()[i].r << " " << mesh->getColors<3>()[i].g << " " << mesh->getColors<3>()[i].b << "\n";
	}
	else if (mesh->hasColorsRgba()) {
		for (int i = 0; i < n; ++i)
			stream << "v " << mesh->getPositions<3>()[i].x << " " << mesh->getPositions<3>()[i].y << " " << mesh->getPositions<3>()[i].z << " " <<
			mesh->getColors<4>()[i].r << " " << mesh->getColors<4>()[i].g << " " << mesh->getColors<4>()[i].b << "\n";
	}
	else {
		for (int i = 0; i < n; ++i)
			stream << "v " << mesh->getPositions<3>()[i].x << " " << mesh->getPositions<3>()[i].y << " " << mesh->getPositions<3>()[i].z << "\n";
	}
	
	if (mesh->hasNormals()) {
		for (int i = 0; i < n; ++i)
			stream << "vn " << mesh->getNormals()[i].x << " " << mesh->getNormals()[i].y << " " << mesh->getNormals()[i].z << "\n";
	}
	if (mesh->hasColors()) {
		for (int i = 0; i < n; ++i)
			stream << "vc " << mesh->getColors<4>()[i].r << " " << mesh->getColors<4>()[i].g << " " << mesh->getColors<4>()[i].b << " " << mesh->getColors<4>()[i].a << "\n";
	}
	if (mesh->hasTexCoords0()) {
		int c = mesh->getAttribDims(cinder::geom::Attrib::TEX_COORD_0);
		for (int i = 0; i < n; ++i) {
			stream << "vt";
			for (int j=0; j<c; ++j)
				stream << " " << mesh->getBufferTexCoords0()[c*i+j];
			stream << "\n";
		}
	}
	if (mesh->hasTexCoords1()) {
		for (int i = 0; i < n; ++i) {
			stream << "vg " << mesh->getTexCoords1<3>()[i].x << " " << mesh->getTexCoords1<3>()[i].y << " " << mesh->getTexCoords1<3>()[i].z << "\n";
		}
	}
	//triangles
	for (int i = 0; i < m; ++i) {
		if (mesh->hasTexCoords0() && mesh->hasNormals()) {
			stream << "f "
				<< (mesh->getIndices()[3 * i] + 1) << "/" << (mesh->getIndices()[3 * i] + 1) << "/" << (mesh->getIndices()[3 * i] + 1) << " "
				<< (mesh->getIndices()[3 * i + 1] + 1) << "/" << (mesh->getIndices()[3 * i + 1] + 1) << "/" << (mesh->getIndices()[3 * i + 1] + 1) << " "
				<< (mesh->getIndices()[3 * i + 2] + 1) << "/" << (mesh->getIndices()[3 * i + 2] + 1) << "/" << (mesh->getIndices()[3 * i + 2] + 1) << " "
				<< "\n";
		}
		else if (mesh->hasTexCoords0()) {
			stream << "f "
				<< (mesh->getIndices()[3 * i] + 1) << "/" << (mesh->getIndices()[3 * i] + 1) << " "
				<< (mesh->getIndices()[3 * i + 1] + 1) << "/" << (mesh->getIndices()[3 * i + 1] + 1) << " "
				<< (mesh->getIndices()[3 * i + 2] + 1) << "/" << (mesh->getIndices()[3 * i + 2] + 1) << " "
				<< "\n";
		}
		else if (mesh->hasNormals()) {
			stream << "f "
				<< (mesh->getIndices()[3 * i] + 1) << "//" << (mesh->getIndices()[3 * i] + 1) << " "
				<< (mesh->getIndices()[3 * i + 1] + 1) << "//" << (mesh->getIndices()[3 * i + 1] + 1) << " "
				<< (mesh->getIndices()[3 * i + 2] + 1) << "//" << (mesh->getIndices()[3 * i + 2] + 1) << " "
				<< "\n";
		}
		else {
			stream << "f " << (mesh->getIndices()[3 * i] + 1) << " " << (mesh->getIndices()[3 * i + 1] + 1) << " " << (mesh->getIndices()[3 * i + 2] + 1) << "\n";
		}
	}

	stream.close();
}

void ObjectLoader::saveMeshPly(ci::TriMeshRef mesh, const std::string & path)
{
	using namespace tinyply;

	//create ply file
	PlyFile file;
	file.add_properties_to_element("vertex", { "x", "y", "z" },
		Type::FLOAT32, mesh->getBufferPositions().size()/3, reinterpret_cast<uint8_t*>(mesh->getBufferPositions().data()), Type::INVALID, 0);
	if (mesh->hasNormals())
		file.add_properties_to_element("vertex", { "nx", "ny", "nz" },
			Type::FLOAT32, mesh->getNormals().size(), reinterpret_cast<uint8_t*>(mesh->getNormals().data()), Type::INVALID, 0);
	if (mesh->hasTexCoords0()) {
		int c = mesh->getAttribDims(cinder::geom::Attrib::TEX_COORD_0);
		if (c==2)
			file.add_properties_to_element("vertex", { "s", "t" },
				Type::FLOAT32, mesh->getBufferTexCoords0().size()/2, reinterpret_cast<uint8_t*>(mesh->getBufferTexCoords0().data()), Type::INVALID, 0);
		else if (c == 3)
			file.add_properties_to_element("vertex", { "gx", "gy", "gz" },
				Type::FLOAT32, mesh->getBufferTexCoords0().size() / 3, reinterpret_cast<uint8_t*>(mesh->getBufferTexCoords0().data()), Type::INVALID, 0);
	}
	file.add_properties_to_element("face", { "vertex_indices" },
		Type::INT32, mesh->getNumIndices()/3, reinterpret_cast<uint8_t*>(mesh->getIndices().data()), Type::UINT8, 3);

	//write file
	std::filebuf fb_binary;
	fb_binary.open(path, std::ios::out | std::ios::binary);
	std::ostream outstream_binary(&fb_binary);
	if (outstream_binary.fail()) throw std::runtime_error("failed to open " + path);
	file.write(outstream_binary, true);
}

void ObjectLoader::savePointsPly(const std::vector<ci::vec3>& points, const std::string & path)
{
}
